"""Random search on 2023 -> 2024, then final models on 2023-2024 scored on 2025.

    python -m src.fit_models                       # all 27 combinations
    python -m src.fit_models --model RF --strategy SMOTE
    python -m src.fit_models --force               # refit even if cached results exist
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

import config
from src.models import combinations, draw_setting, fit_and_score
from src.reproducibility import make_manifest, make_seed, manifests_match, setup_logging, worker_logging

log = logging.getLogger("fit")


def load_analytic(output_dir=config.OUTPUT_DIR, name="analytic.pkl"):
    path = Path(output_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.prepare_data` first")
    return pd.read_pickle(path), path


NOT_PREDICTORS = set(config.COHORT_COLUMNS + config.AUXILIARY_COLUMNS + config.ID_COLUMNS + ["y"])


def check_predictors(features):
    """Refuse cohort-only, auxiliary or identifier columns as predictors.

    LAST_BILLED, for example, is built from billing after the cut-off and would leak the
    outcome. SCHOOL_EXIT_RATE_PAST is allowed only as a replacement in a robustness run.
    """
    bad = sorted(set(features) & NOT_PREDICTORS - {"SCHOOL_EXIT_RATE_PAST"})
    if bad:
        raise ValueError(f"not allowed as predictors: {bad}")


def split(data, train_years, test_years, features=config.FEATURES):
    check_predictors(features)
    train = data[data["ANO"].isin(train_years)]
    test = data[data["ANO"].isin(test_years)]
    return train[features].to_numpy(), train["y"].to_numpy(), test[features].to_numpy(), test["y"].to_numpy()


def _paths(out_dir, model, strategy):
    out_dir = Path(out_dir)
    return (out_dir / "tuning" / f"{model}_{strategy}.json",
            out_dir / "predictions" / f"{model}_{strategy}.npz",
            out_dir / "manifests" / f"{model}_{strategy}.json")


def run_combination(model, strategy, data, analytic_path, out_dir, features, n_settings, force,
                    base_seed=config.BASE_SEED):
    """Search and final fit for one combination. Returns a status record."""
    worker_logging(Path(out_dir) / "logs", "fit")
    tuning_file, pred_file, manifest_file = _paths(out_dir, model, strategy)
    manifest = make_manifest(analytic_path, model=model, strategy=strategy, features=features,
                             n_settings=n_settings, seeds=config.SEEDS, base_seed=base_seed,
                             years={"search": [config.TRAIN_YEAR, config.VALIDATION_YEAR],
                                    "final_train": config.FINAL_TRAIN_YEARS, "test": config.TEST_YEAR})

    if pred_file.exists() and manifest_file.exists() and not force:
        diff = manifests_match(json.loads(manifest_file.read_text()), manifest)
        if not diff:
            return {"model": model, "strategy": strategy, "status": "cached"}
        raise RuntimeError(f"{model}/{strategy}: cached results were produced with different {diff}; "
                           "rerun with --force to replace them")

    log.info("Fitting %s with %s", model, strategy)
    start = time.time()
    X_tr, y_tr, X_val, y_val = split(data, [config.TRAIN_YEAR], [config.VALIDATION_YEAR], features)
    seed = make_seed(model, strategy, stage="search_fit", base_seed=base_seed)
    settings = []
    for i in range(n_settings):
        params = draw_setting(model, strategy, i, base_seed)
        try:
            scores = fit_and_score(model, strategy, params, seed, X_tr, y_tr, X_val)
            settings.append({"index": i, "params": params, "val_ap": average_precision_score(y_val, scores),
                             "status": "ok"})
        except Exception as exc:  # recorded, never turned into a score
            log.error("%s/%s setting %d %s seed %d failed: %s: %s", model, strategy, i, params, seed,
                      type(exc).__name__, exc)
            settings.append({"index": i, "params": params, "val_ap": None, "status": "failed",
                             "error": f"{type(exc).__name__}: {exc}"})
    ok = [s for s in settings if s["status"] == "ok"]
    if not ok:
        raise RuntimeError(f"{model}/{strategy}: every search setting failed")
    best = max(ok, key=lambda s: s["val_ap"])

    X_tr, y_tr, X_te, y_te = split(data, config.FINAL_TRAIN_YEARS, [config.TEST_YEAR], features)
    scores, secs = [], []
    for rep in config.SEEDS:
        t0 = time.time()
        final_seed = make_seed(model, strategy, stage="final_fit", repetition=rep, base_seed=base_seed)
        scores.append(fit_and_score(model, strategy, best["params"], final_seed, X_tr, y_tr, X_te))
        secs.append(time.time() - t0)

    for p in (tuning_file, pred_file, manifest_file):
        p.parent.mkdir(parents=True, exist_ok=True)
    tuning_file.write_text(json.dumps({"model": model, "strategy": strategy, "best": best,
                                       "settings": settings}, indent=1, default=str))
    np.savez(pred_file, scores=np.array(scores), secs=np.array(secs), y=y_te)
    manifest["best_params"] = best["params"]
    manifest_file.write_text(json.dumps(manifest, indent=1, default=str))

    log.info("%s/%s done in %.0fs: validation PR-AUC %.3f, %d/%d settings ok",
             model, strategy, time.time() - start, best["val_ap"], len(ok), n_settings)
    return {"model": model, "strategy": strategy, "status": "fitted",
            "failed_settings": n_settings - len(ok)}


def _safe_run(*args):
    try:
        return run_combination(*args)
    except Exception as exc:
        log.error("%s/%s failed: %s", args[0], args[1], exc)
        return {"model": args[0], "strategy": args[1], "status": "failed", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=config.MODELS)
    parser.add_argument("--strategy", choices=config.STRATEGIES + ["built-in"])
    parser.add_argument("--n-settings", type=int, default=config.N_SETTINGS)
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--seed", type=int, default=config.BASE_SEED,
                        help="base seed; changing it gives a different (non-reproduced) run")
    parser.add_argument("--feature-set", choices=list(config.FEATURE_SETS), default="full")
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    setup_logging(out_dir / "logs", args.log_level, "fit")
    data, analytic_path = load_analytic(out_dir)
    features = config.FEATURE_SETS[args.feature_set]
    if args.feature_set != "full":
        out_dir = out_dir / "feature_sets" / args.feature_set
        log.info("Using predictor set %s (%d variables); results go to %s", args.feature_set, len(features), out_dir)

    jobs = [(m, s) for m, s in combinations()
            if (args.model is None or m == args.model) and (args.strategy is None or s == args.strategy)]
    if not jobs:
        parser.error("no combination matches --model/--strategy")
    jobs.sort(key=lambda j: j[0] != "RF")   # the slowest first, so the workers finish together
    log.info("%d combinations, %d settings each, seeds %s", len(jobs), args.n_settings, config.SEEDS)

    status = Parallel(n_jobs=args.n_jobs)(
        delayed(_safe_run)(m, s, data, analytic_path, out_dir, features, args.n_settings, args.force, args.seed)
        for m, s in jobs)

    summary = pd.DataFrame(status)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "logs" / "fit_summary.csv", index=False)
    counts = summary["status"].value_counts().to_dict()
    log.info("Requested %d, fitted %d, cached %d, failed %d", len(jobs), counts.get("fitted", 0),
             counts.get("cached", 0), counts.get("failed", 0))
    for row in summary[summary["status"] == "failed"].itertuples():
        log.error("Failed: %s/%s: %s", row.model, row.strategy, row.error)


if __name__ == "__main__":
    main()
