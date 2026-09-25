"""Exploratory analyses. None of them is used to choose the models of the main analysis.

    python -m src.extra_analyses                  # baseline, list sizes, stability, schools,
                                                  # overlap, recalibration
    python -m src.extra_analyses --robustness     # also refit with alternative predictor sets
                                                  # and the past-only school exit rate

Hyperparameters are always the ones chosen in the main analysis.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare, kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config
from src import figures
from src.build_results import load_predictions, select_strategies
from src.fit_models import load_analytic, split
from src.metrics import (calculate_calibration_metrics, expected_hits, list_size, recall_at_k,
                         top_k_indices)
from src.models import combinations, fit_and_score, has_probabilities
from src.reproducibility import make_seed, setup_logging, worker_logging
from src.statistics import bootstrap_winner_frequency, stratified_bootstrap_indices

log = logging.getLogger("extra")


def additive_baseline(train, test):
    """Four adverse signals standardised with the training years and capped at +/- 3 SD."""
    def signals(df):
        return np.column_stack([-df["PCT_T2"], df["DECLINE_PCT"].fillna(0), df["T2_BELOW6"], df["ARREARS_TO_AUG"]])
    mu, sd = signals(train).mean(axis=0), signals(train).std(axis=0)
    return np.clip((signals(test) - mu) / sd, -3, 3).sum(axis=1)


def metrics_by_list_size(y, scores):
    rows = {}
    for name, s in scores.items():
        row = {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s)}
        for share in config.LIST_SHARES:
            row[f"Recall top {share:.1%}"] = recall_at_k(y, s, list_size(len(y), share))
        rows[name] = row
    table = pd.DataFrame(rows).T
    return table, table.rank(ascending=False, method="min").astype(int)


def bootstrap_first_place(y, scores):
    names = list(scores)
    metrics = {"PR-AUC": average_precision_score, "ROC-AUC": roc_auc_score}
    for share in (0.01, 0.05, 0.10):
        metrics[f"Recall top {share:.1%}"] = lambda yb, sb, sh=share: recall_at_k(yb, sb, list_size(len(yb), sh))
    values = {m: [] for m in metrics}
    base = {"ap": [], "auc": [], "rec5": []}
    ref = {"LR": {"ap": [], "auc": [], "rec5": []}, "RF": {"ap": [], "auc": [], "rec5": []}}
    for idx in stratified_bootstrap_indices(y, config.N_BOOT, config.BOOTSTRAP_SEED):
        yb = y[idx]
        for metric, fn in metrics.items():
            values[metric].append([fn(yb, scores[n][idx]) for n in names])
        for store, who in [(base, "Baseline"), (ref["LR"], "LR"), (ref["RF"], "RF")]:
            sb = scores[who][idx]
            store["ap"].append(average_precision_score(yb, sb))
            store["auc"].append(roc_auc_score(yb, sb))
            store["rec5"].append(recall_at_k(yb, sb, list_size(len(yb), 0.05)))
    first = pd.DataFrame({m: bootstrap_winner_frequency(np.array(v), names) for m, v in values.items()})
    baseline = {}
    for metric in base:
        b = np.array(base[metric])
        baseline[metric] = {"low": np.quantile(b, 0.025), "high": np.quantile(b, 0.975)}
        for who in ("LR", "RF"):
            d = b - np.array(ref[who][metric])
            baseline[metric][f"diff_vs_{who}_low"] = np.quantile(d, 0.025)
            baseline[metric][f"diff_vs_{who}_high"] = np.quantile(d, 0.975)
    return first, baseline


def validation_scores(data, tuning, out_dir, n_jobs):
    """Refit every combination (search seed) on 2023 -> 2024 and on 2024 -> 2023."""
    def run(m, s, tr, va):
        worker_logging(out_dir / "logs", "extra")
        X_tr, y_tr, X_va, _ = split(data, [tr], [va])
        return m, s, tr, va, fit_and_score(m, s, tuning[(m, s)]["best"]["params"],
                                           make_seed(m, s, stage="search_fit"), X_tr, y_tr, X_va)
    jobs = [(m, s, a, b) for m, s in combinations() for a, b in [(2023, 2024), (2024, 2023)]]
    jobs.sort(key=lambda j: j[0] != "RF")
    log.info("%d validation refits", len(jobs))
    res = Parallel(n_jobs=n_jobs)(delayed(run)(*j) for j in jobs)
    return {(m, s, a, b): sc for m, s, a, b, sc in res}


def strategy_stability(data, val, ensemble):
    y24 = data[data["ANO"] == config.VALIDATION_YEAR]["y"].to_numpy()
    y23 = data[data["ANO"] == 2023]["y"].to_numpy()
    y25 = data[data["ANO"] == config.TEST_YEAR]["y"].to_numpy()
    rows = [{"model": m, "strategy": s,
             "val_2024": average_precision_score(y24, val[(m, s, 2023, 2024)]),
             "val_2023": average_precision_score(y23, val[(m, s, 2024, 2023)]),
             "test_2025": average_precision_score(y25, ensemble[(m, s)])} for m, s in combinations()]
    stab = pd.DataFrame(rows)
    stab["val_mean"] = stab[["val_2024", "val_2023"]].mean(axis=1)

    choices = []
    for m in config.GENERAL_MODELS:
        part = stab[stab["model"] == m].set_index("strategy")
        row = {"model": m}
        for col in ("val_2024", "val_2023", "val_mean"):
            pick = part[col].idxmax()
            row[f"chosen_{col}"] = pick
            row[f"test_if_{col}"] = part.loc[pick, "test_2025"]
        row["oracle_strategy"] = part["test_2025"].idxmax()
        row["oracle_test"] = part["test_2025"].max()
        choices.append(row)
    choices = pd.DataFrame(choices)
    choices["selection_cost_val_2024"] = choices["oracle_test"] - choices["test_if_val_2024"]

    rng = np.random.default_rng(24)
    pos, neg = np.flatnonzero(y24 == 1), np.flatnonzero(y24 == 0)
    freq = {m: dict.fromkeys(config.STRATEGIES, 0) for m in config.GENERAL_MODELS}
    for _ in range(config.N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for m in config.GENERAL_MODELS:
            aps = {s: average_precision_score(y24[idx], val[(m, s, 2023, 2024)][idx]) for s in config.STRATEGIES}
            freq[m][max(aps, key=aps.get)] += 1
    freq = pd.DataFrame(freq).T / config.N_BOOT

    corr = {f"spearman_{a}_{b}": stab[a].corr(stab[b], method="spearman")
            for a, b in [("val_2024", "val_2023"), ("val_2024", "test_2025"), ("val_2023", "test_2025"),
                         ("val_mean", "test_2025")]}
    return stab, choices, freq, corr


def school_comparison(y, schools, scores):
    eligible = [s for s in np.unique(schools) if y[schools == s].sum() >= config.MIN_SCHOOL_POSITIVES]
    ap = pd.DataFrame(index=eligible, columns=list(scores), dtype=float)
    rec = ap.copy()
    for s in eligible:
        mask = schools == s
        ys = y[mask]
        k = max(1, list_size(mask.sum(), config.MAIN_LIST_SHARE))
        for name, sc in scores.items():
            ap.loc[s, name] = average_precision_score(ys, sc[mask])
            rec.loc[s, name] = expected_hits(ys, sc[mask], k) / ys.sum()
    fr_ap = friedmanchisquare(*[ap[c] for c in ap.columns])
    fr_rec = friedmanchisquare(*[rec[c] for c in rec.columns])
    nemenyi = sp.posthoc_nemenyi_friedman(ap.to_numpy())
    nemenyi.index = nemenyi.columns = list(scores)
    summary = {"n_schools": len(eligible), "n_departures": int(sum(y[schools == s].sum() for s in eligible)),
               "friedman_ap": {"chi2": fr_ap.statistic, "p": fr_ap.pvalue},
               "friedman_recall_top5": {"chi2": fr_rec.statistic, "p": fr_rec.pvalue},
               "mean_rank_ap": ap.rank(axis=1, ascending=False).mean().to_dict(),
               "mean_ap": ap.mean().to_dict(), "mean_recall_top5": rec.mean().to_dict()}
    return ap, nemenyi, summary


def list_overlap(y, scores, chosen):
    k = list_size(len(y), config.MAIN_LIST_SHARE)
    lists = {n: set(top_k_indices(s, k, make_seed(n, chosen.get(n, "-"), 50, "tie"))) for n, s in scores.items()}
    names = list(scores)
    jac = pd.DataFrame([[len(lists[a] & lists[b]) / len(lists[a] | lists[b]) for b in names] for a in names],
                       index=names, columns=names)
    models = [n for n in names if n != "Baseline"]
    left = set(np.flatnonzero(y == 1))
    union, inter = set().union(*[lists[m] for m in models]), set.intersection(*[lists[m] for m in models])
    caught = {i: sum(i in lists[m] for m in models) for i in left}
    off = jac.loc[models, models].where(~np.eye(len(models), dtype=bool))
    return jac, {"union_size": len(union), "union_hits": len(union & left),
                 "intersection_size": len(inter), "intersection_hits": len(inter & left),
                 "caught_by_none": sum(v == 0 for v in caught.values()),
                 "caught_by_all": sum(v == len(models) for v in caught.values()),
                 "jaccard_min_models": float(off.min().min()), "jaccard_max_models": float(off.max().max())}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def recalibration(data, val, ensemble, chosen):
    """Platt recalibration fitted on 2024 validation scores and applied to 2025.

    The calibrator is estimated on a model trained on 2023 and applied to the final model
    trained on 2023-2024, so it can only correct a shift that both share. 2025 is never used
    for fitting.
    """
    y24 = data[data["ANO"] == config.VALIDATION_YEAR]["y"].to_numpy()
    y25 = data[data["ANO"] == config.TEST_YEAR]["y"].to_numpy()
    rows = []
    for m, s in combinations():
        if not has_probabilities(m) or not (s == "none" or s == chosen[m]):
            continue
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(val[(m, s, 2023, 2024)]), y24)
        recal = cal.predict_proba(_logit(ensemble[(m, s)]))[:, 1]
        for label, p in (("original", ensemble[(m, s)]), ("platt_2024", recal)):
            rows.append({"model": m, "strategy": s, "version": label, "brier": brier_score_loss(y25, p),
                         "ap": average_precision_score(y25, p), **calculate_calibration_metrics(y25, p)})
    return pd.DataFrame(rows)


def robustness(data_by_variant, tuning, chosen, n_jobs, out_dir):
    """Refit the chosen pipelines (fixed hyperparameters, five seeds) on alternative inputs."""
    def run(variant, m, rep):
        worker_logging(out_dir / "logs", "extra")
        data, features = data_by_variant[variant]
        X_tr, y_tr, X_te, _ = split(data, config.FINAL_TRAIN_YEARS, [config.TEST_YEAR], features)
        return variant, m, fit_and_score(m, chosen[m], tuning[(m, chosen[m])]["best"]["params"],
                                         make_seed(m, chosen[m], stage="final_fit", repetition=rep), X_tr, y_tr, X_te)
    jobs = [(v, m, r) for v in data_by_variant for m in config.MODELS for r in config.SEEDS]
    jobs.sort(key=lambda j: j[1] != "RF")
    log.info("%d robustness refits", len(jobs))
    res = Parallel(n_jobs=n_jobs)(delayed(run)(*j) for j in jobs)
    collected = {}
    for v, m, sc in res:
        collected.setdefault((v, m), []).append(sc)
    rows = []
    for (v, m), sc in collected.items():
        data = data_by_variant[v][0]
        y = data[data["ANO"] == config.TEST_YEAR]["y"].to_numpy()
        s = np.mean(sc, axis=0)
        rows.append({"variant": v, "model": m, "n_features": len(data_by_variant[v][1]),
                     "ap": average_precision_score(y, s), "auc": roc_auc_score(y, s),
                     "recall_top5": recall_at_k(y, s, list_size(len(y), config.MAIN_LIST_SHARE))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    out = Path(args.output_dir)
    tables, figs = out / "tables", out / "figures"
    setup_logging(out / "logs", args.log_level, "extra")

    data, analytic_path = load_analytic(out)
    seed_scores, tuning = load_predictions(out, analytic_path)
    ensemble = {k: v.mean(axis=0) for k, v in seed_scores.items()}
    chosen = select_strategies(tuning)
    train = data[data["ANO"].isin(config.FINAL_TRAIN_YEARS)]
    test = data[data["ANO"] == config.TEST_YEAR].reset_index(drop=True)
    y = test["y"].to_numpy()
    scores = {m: ensemble[(m, chosen[m])] for m in config.MODELS}
    scores["Baseline"] = additive_baseline(train, test)

    by_size, ranks = metrics_by_list_size(y, scores)
    by_size.to_csv(tables / "extra_metrics_by_list_size.csv")
    ranks.to_csv(tables / "extra_ranks_by_metric.csv")
    taus = {c: kendalltau(by_size["PR-AUC"], by_size[c]).statistic for c in by_size.columns}
    first, baseline = bootstrap_first_place(y, scores)
    first.to_csv(tables / "extra_share_bootstrap_first_by_metric.csv")

    val = validation_scores(data, tuning, out, args.n_jobs)
    stab, choices, freq, corr = strategy_stability(data, val, ensemble)
    stab.to_csv(tables / "extra_validation_years.csv", index=False)
    choices.to_csv(tables / "extra_strategy_choice.csv", index=False)
    freq.to_csv(tables / "extra_bootstrap_choice_2024.csv")

    school_ap, nemenyi, school = school_comparison(y, test["SCHOOL_ID"].to_numpy(), scores)
    school_ap.to_csv(tables / "extra_school_prauc.csv", index_label="SCHOOL_ID")
    nemenyi.to_csv(tables / "extra_school_nemenyi.csv")

    jaccard, overlap = list_overlap(y, scores, chosen)
    jaccard.to_csv(tables / "extra_jaccard_top5.csv")

    recal = recalibration(data, val, ensemble, chosen)
    recal.to_csv(tables / "extra_recalibration.csv", index=False)

    summary = {"chosen": chosen, "baseline": baseline, "kendall_tau_vs_prauc": taus,
               "rank_agreement": corr, "schools": school, "overlap": overlap}
    (tables / "extra_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    figures.plot_rank_heatmap(ranks, figs)
    figures.plot_school_cd(school_ap.rank(axis=1, ascending=False).mean(), nemenyi, school["n_schools"], figs)
    figures.plot_overlap(jaccard, figs)

    if args.robustness:
        variants = {name: (data, feats) for name, feats in config.FEATURE_SETS.items() if name != "full"}
        past = config.FEATURES.copy()
        past[past.index("SCHOOL_EXIT_RATE")] = "SCHOOL_EXIT_RATE_PAST"
        variants["school_exit_rate_past_only"] = (data, past)
        rob = robustness(variants, tuning, chosen, args.n_jobs, out)
        full = pd.DataFrame([{"variant": "full", "model": m, "n_features": len(config.FEATURES),
                              "ap": average_precision_score(y, scores[m]), "auc": roc_auc_score(y, scores[m]),
                              "recall_top5": recall_at_k(y, scores[m], list_size(len(y), config.MAIN_LIST_SHARE))}
                             for m in config.MODELS])
        pd.concat([full, rob]).to_csv(tables / "extra_robustness.csv", index=False)
    log.info("Exploratory tables written to %s", tables)


if __name__ == "__main__":
    main()
