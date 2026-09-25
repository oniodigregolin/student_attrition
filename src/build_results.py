"""Main results: tables, tests and figures 2, 4, 6 and 7.

The main analysis uses, for each algorithm, the strategy with the best 2024 validation
PR-AUC. Anything that looks at which strategy was best on 2025 is labelled
"oracle performance" (not available during model selection).

python -m src.build_results
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from src import figures
from src.fit_models import load_analytic
from src.metrics import (calculate_calibration_metrics, calculate_default_threshold_metrics,
    calculate_model_metrics, list_size, recall_at_k, top_k_confusion)
from src.models import combinations, has_probabilities
from src.reproducibility import file_hash, object_hash, setup_logging, study_settings
from src.statistics import (bootstrap_winner_frequency, friedman_strategy_test, paired_bootstrap_comparison,
    stratified_bootstrap_indices, validation_test_correlation)

log = logging.getLogger("results")

def load_predictions(out_dir, analytic_path):
    """Seed-level scores for every combination, after checking that all manifests agree."""
    expected = {"analytic_hash": file_hash(analytic_path), "config_hash": object_hash(study_settings())}
    scores, tuning = {}, {}
    for model, strategy in combinations():
        manifest = json.loads((out_dir / "manifests" / f"{model}_{strategy}.json").read_text())
        bad = [k for k, v in expected.items() if manifest.get(k) != v]
        if bad:
            raise RuntimeError(f"{model}/{strategy} was fitted with a different {bad}; refit with --force")
        scores[(model, strategy)] = np.load(out_dir / "predictions" / f"{model}_{strategy}.npz")["scores"]
        tuning[(model, strategy)] = json.loads((out_dir / "tuning" / f"{model}_{strategy}.json").read_text())
    return scores, tuning

def select_strategies(tuning):
    """Strategy with the highest 2024 validation PR-AUC for each model.

    Ties go to the strategy listed first in ``config.STRATEGIES`` (as ``idxmax`` would do).
    """
    chosen = {}
    for model, strategy in combinations():
        score = tuning[(model, strategy)]["best"]["val_ap"]
        if model not in chosen or score > tuning[(model, chosen[model])]["best"]["val_ap"]:
            chosen[model] = strategy
    return chosen

def build(out_dir=config.OUTPUT_DIR):
    out_dir = Path(out_dir)
    tables, figs = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    data, analytic_path = load_analytic(out_dir)
    test = data[data["ANO"] == config.TEST_YEAR]
    y = test["y"].to_numpy()
    train_prev = data[data["ANO"].isin(config.FINAL_TRAIN_YEARS)]["y"].mean()
    seed_scores, tuning = load_predictions(out_dir, analytic_path)

    # metrics per seed and metrics of the seed-averaged ensemble are kept apart
    per_seed, ensemble, rows = [], {}, []
    for (m, s), sc in seed_scores.items():
        prob = has_probabilities(m)
        for rep, one in zip(config.SEEDS, sc):
            per_seed.append({"model": m, "strategy": s, "seed": rep,
                **calculate_model_metrics(y, one, prob, train_prev)})
        ensemble[(m, s)] = sc.mean(axis=0)
        default = calculate_default_threshold_metrics(y, ensemble[(m, s)], prob)
        rows.append({"model": m, "strategy": s, "val_ap": tuning[(m, s)]["best"]["val_ap"],
            **calculate_model_metrics(y, ensemble[(m, s)], prob, train_prev),
            "default_flagged": default["flagged"], "default_tp": default["tp"]})
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(tables / "metrics_per_seed.csv", index=False)
    combos = pd.DataFrame(rows)
    seed_mean = per_seed.groupby(["model", "strategy"])[["ap", "auc"]].mean().add_suffix("_mean_of_seeds")
    # population SD over the five seeds (ddof=0)
    seed_sd = per_seed.groupby(["model", "strategy"])[["ap"]].std(ddof=0).add_suffix("_sd_of_seeds")
    combos = combos.merge(seed_mean, on=["model", "strategy"]).merge(seed_sd, on=["model", "strategy"])
    combos.to_csv(tables / "metrics_ensemble_all_combinations.csv", index=False)

    chosen = select_strategies(tuning)
    oracle = combos[combos["model"].isin(config.GENERAL_MODELS)].sort_values("ap", ascending=False) \
        .groupby("model").head(1)[["model", "strategy", "ap"]].rename(
        columns={"strategy": "oracle_strategy", "ap": "oracle_ap"})
    selection = pd.DataFrame({"model": list(chosen), "chosen_strategy": list(chosen.values())})
    selection["chosen_ap"] = [combos.set_index(["model", "strategy"]).loc[(m, s), "ap"] for m, s in chosen.items()]
    selection = selection.merge(oracle, on="model", how="left")
    selection["selection_cost"] = selection["oracle_ap"] - selection["chosen_ap"]
    selection["note"] = "oracle performance: retrospective test performance not available during model selection"
    selection.to_csv(tables / "selected_strategies_and_oracle.csv", index=False)
    log.info("Chosen on validation: %s", chosen)

    scores = {m: ensemble[(m, chosen[m])] for m in config.MODELS}

    # confusion matrices: default threshold and one concrete top-5% list
    cm_rows, flagged = [], {}
    k1 = list_size(len(y), config.SHORT_LIST_SHARE)
    for m in config.MODELS:
        default = calculate_default_threshold_metrics(y, scores[m], has_probabilities(m))
        top5, flagged[m] = top_k_confusion(y, scores[m], config.MAIN_LIST_SHARE, m, chosen[m])
        cm_rows.append({"model": m, "strategy": chosen[m], **{f"default_{k}": v for k, v in default.items()},
            **{f"top5_list_{k}": v for k, v in top5.items()},
            "top5_expected_recall": recall_at_k(y, scores[m], list_size(len(y), config.MAIN_LIST_SHARE)),
            "top1_expected_hits": recall_at_k(y, scores[m], k1) * y.sum()})
    cm = pd.DataFrame(cm_rows)
    cm.to_csv(tables / "confusion_matrices.csv", index=False)

    # bootstrap on the test set (predictions fixed)
    k5 = list_size(len(y), config.MAIN_LIST_SHARE)
    boot = {key: {m: [] for m in config.MODELS} for key in ("ap", "auc", "recall_top5")}
    for b, idx in enumerate(stratified_bootstrap_indices(y, config.N_BOOT, config.BOOTSTRAP_SEED)):
        yb = y[idx]
        for m in config.MODELS:
            sb = scores[m][idx]
            boot["ap"][m].append(average_precision_score(yb, sb))
            boot["auc"][m].append(roc_auc_score(yb, sb))
            boot["recall_top5"][m].append(recall_at_k(yb, sb, k5))
        if (b + 1) % 250 == 0:
            log.info("Bootstrap %d/%d", b + 1, config.N_BOOT)
    pd.concat({k: pd.DataFrame(v) for k, v in boot.items()}, names=["metric", "sample"]) \
        .to_csv(tables / "bootstrap_samples.csv")

    main = combos.set_index(["model", "strategy"]).loc[[(m, chosen[m]) for m in config.MODELS]].reset_index()
    main = main.set_index("model")
    comparisons = []
    for key in ("ap", "auc", "recall_top5"):
        main[key + "_low"] = [np.quantile(boot[key][m], 0.025) for m in main.index]
        main[key + "_high"] = [np.quantile(boot[key][m], 0.975) for m in main.index]
        comp = paired_bootstrap_comparison(boot[key], "LR").assign(metric=key)
        comp["diff"] = [main.loc[m, key] - main.loc["LR", key] for m in comp["model"]]
        comparisons.append(comp)
    comparisons = pd.concat(comparisons)
    comparisons.to_csv(tables / "paired_differences_vs_logistic_regression.csv", index=False)
    winners = bootstrap_winner_frequency(np.column_stack([boot["ap"][m] for m in config.MODELS]), config.MODELS)
    main["share_bootstrap_highest_prauc"] = winners
    main.to_csv(tables / "main_results.csv")

    # strategies: algorithms as blocks, therefore exploratory
    tests = {"note": "Friedman tests use algorithms as blocks and are exploratory"}
    general = combos[combos["model"].isin(config.GENERAL_MODELS)]
    for metric in ("ap", "val_ap", "recall_top5", "brier"):
        table = general.pivot(index="model", columns="strategy", values=metric)[config.STRATEGIES].dropna()
        tests[metric] = friedman_strategy_test(-table if metric == "brier" else table)
    tests["validation_vs_test"] = validation_test_correlation(combos["val_ap"], combos["ap"])
    (tables / "tests.json").write_text(json.dumps(tests, indent=1, default=float))

    calib = [{"model": m, "strategy": s, **calculate_calibration_metrics(y, p),
        "brier": combos.set_index(["model", "strategy"]).loc[(m, s), "brier"],
        "brier_skill": combos.set_index(["model", "strategy"]).loc[(m, s), "brier_skill"]}
        for (m, s), p in ensemble.items() if has_probabilities(m)]
    pd.DataFrame(calib).to_csv(tables / "calibration.csv", index=False)

    prevalence = y.mean()
    figures.plot_performance_intervals(main, chosen, prevalence, figs)
    figures.plot_confusion_matrices(y, flagged, config.MODELS, figs)
    ap_t = general.pivot(index="model", columns="strategy", values="ap").loc[config.GENERAL_MODELS, config.STRATEGIES]
    pr_t = general[general["model"] != "SVM"].pivot(index="model", columns="strategy", values="mean_prob") \
        .reindex(index=config.GENERAL_MODELS, columns=config.STRATEGIES) * 100
    figures.plot_strategy_heatmaps(ap_t, pr_t, figs)
    rarity = tables / "rarity_summary.csv"
    if rarity.exists():
        figures.plot_rarity_experiment(pd.read_csv(rarity), config.MODELS, prevalence, figs)
    else:
        log.warning("rarity_summary.csv not found; run src.rarity_experiment to draw figure 7")
    log.info("Tables in %s, figures in %s", tables, figs)
    return main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(Path(args.output_dir) / "logs", args.log_level, "results")
    build(args.output_dir)

if __name__ == "__main__":
    main()
