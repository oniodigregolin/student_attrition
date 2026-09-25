"""Refit each model keeping only part of the training departures (100/50/25/10%).

Every model keeps the strategy and hyperparameters chosen on the 2024 validation data;
all students who stayed are kept and the test year (2025) is fixed. Results are written
row by row, so an interrupted run can be resumed.

At the 100% level all positives are kept, so the only difference from the main results
is that each repetition uses a single model seed (0-9) instead of the average of five.

    python -m src.rarity_experiment
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from src.fit_models import load_analytic, split
from src.metrics import list_size, recall_at_k
from src.build_results import select_strategies
from src.models import combinations, fit_and_score
from src.reproducibility import file_hash, make_seed, object_hash, setup_logging, study_settings, worker_logging

log = logging.getLogger("rarity")


def chosen_settings(out_dir):
    """Strategy with the best 2024 validation PR-AUC for each model, and its parameters."""
    tuning = {}
    for model, strategy in combinations():
        path = Path(out_dir, "tuning", f"{model}_{strategy}.json")
        if not path.exists():
            raise RuntimeError(f"{path} not found; run src.fit_models first")
        tuning[(model, strategy)] = json.loads(path.read_text())
    return {m: tuning[(m, s)] for m, s in select_strategies(tuning).items()}


def positive_subset(positives, level, rep):
    # Use the same positive subset across models to keep rarity comparisons paired.
    rng = np.random.default_rng(make_seed("", "", int(level * 100), "rarity_subset", rep))
    return rng.choice(positives, int(round(level * len(positives))), replace=False)


def one_run(model, setting, level, rep, X_train, y_train, X_test, y_test, log_dir):
    worker_logging(log_dir, "rarity")
    params = dict(setting["best"]["params"])
    strategy = setting["strategy"]
    pos, neg = np.flatnonzero(y_train == 1), np.flatnonzero(y_train == 0)
    rows = np.concatenate([neg, positive_subset(pos, level, rep)])
    X, y = X_train[rows], y_train[rows]
    record = {"model": model, "strategy": strategy, "level": level, "rep": rep, "n_pos": int(y.sum()),
              "prevalence": float(y.mean()), "adjustment": ""}

    if y.sum() < 2:
        return record | {"status": "failed", "error": "fewer than 2 positives"}
    if strategy == "SMOTE" and params["k"] >= y.sum():
        record["adjustment"] = f"SMOTE k_neighbors {params['k']} -> {int(y.sum()) - 1}"
        params["k"] = int(y.sum()) - 1
    for key in ("ratio_s", "ratio"):
        if key in params and params[key] < y.sum() / (len(y) - y.sum()):
            record["adjustment"] += f" {key} below the current ratio"
    try:
        s = fit_and_score(model, strategy, params, make_seed(model, strategy, stage="rarity_fit", repetition=rep),
                          X, y, X_test)
    except Exception as exc:
        return record | {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return record | {"status": "ok", "ap": average_precision_score(y_test, s), "auc": roc_auc_score(y_test, s),
                     "recall_top5": recall_at_k(y_test, s, list_size(len(y_test), config.MAIN_LIST_SHARE))}


def summarise(results):
    ok = results[results["status"] == "ok"]
    g = ok.groupby(["model", "level"])
    summary = g.agg(n_pos=("n_pos", "mean"), prevalence=("prevalence", "mean"), runs=("ap", "size"),
                    ap_mean=("ap", "mean"), ap_sd=("ap", "std"), ap_median=("ap", "median"),
                    ap_q1=("ap", lambda s: s.quantile(0.25)), ap_q3=("ap", lambda s: s.quantile(0.75)),
                    auc_mean=("auc", "mean"), recall_top5_mean=("recall_top5", "mean"),
                    recall_top5_sd=("recall_top5", "std")).reset_index()
    full = summary[summary["level"] == 1.0].set_index("model")["ap_mean"]
    summary["ap_retained"] = summary["ap_mean"] / summary["model"].map(full)
    failures = results[results["status"] != "ok"].groupby(["model", "level"]).size().rename("failed")
    return summary.merge(failures, on=["model", "level"], how="left").fillna({"failed": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--reps", type=int, default=config.RARITY_REPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    out = Path(args.output_dir)
    setup_logging(out / "logs", args.log_level, "rarity")
    data, analytic_path = load_analytic(out)
    settings = chosen_settings(out)
    X_train, y_train, X_test, y_test = split(data, config.FINAL_TRAIN_YEARS, [config.TEST_YEAR])

    rows_file = out / "tables" / "rarity_runs.jsonl"
    manifest_file = out / "manifests" / "rarity.json"
    manifest = {"analytic_hash": file_hash(analytic_path), "config_hash": object_hash(study_settings()),
                "levels": config.RARITY_LEVELS, "reps": args.reps,
                "chosen": {m: [s["strategy"], s["best"]["params"]] for m, s in settings.items()}}
    manifest = json.loads(json.dumps(manifest, default=str))
    if rows_file.exists() and not args.force:
        old = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
        if old != manifest:
            raise RuntimeError("existing rarity results were produced with other settings; use --force")
    else:
        rows_file.parent.mkdir(parents=True, exist_ok=True)
        rows_file.write_text("")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=1))

    done = set()
    for line in rows_file.read_text().splitlines():
        r = json.loads(line)
        done.add((r["model"], r["level"], r["rep"]))
    jobs = [(m, lv, rep) for m in config.MODELS for lv in config.RARITY_LEVELS for rep in range(args.reps)
            if (m, lv, rep) not in done]
    log.info("%d runs to do, %d already saved", len(jobs), len(done))

    runner = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")
    for record in runner(delayed(one_run)(m, settings[m], lv, rep, X_train, y_train, X_test, y_test, out / "logs")
                         for m, lv, rep in jobs):
        with open(rows_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        if record["adjustment"]:
            log.warning("%s level %.2f rep %d: %s", record["model"], record["level"], record["rep"],
                        record["adjustment"].strip())
        if record["status"] != "ok":
            log.error("%s level %.2f rep %d failed: %s", record["model"], record["level"], record["rep"],
                      record["error"])

    results = pd.DataFrame([json.loads(l) for l in rows_file.read_text().splitlines()])
    results.to_csv(out / "tables" / "rarity_runs.csv", index=False)
    summary = summarise(results)
    summary.to_csv(out / "tables" / "rarity_summary.csv", index=False)
    log.info("Rarity: %d runs ok, %d failed", (results["status"] == "ok").sum(), (results["status"] != "ok").sum())


if __name__ == "__main__":
    main()
