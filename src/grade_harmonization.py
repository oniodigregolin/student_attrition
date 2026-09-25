"""Compare the two grade harmonizations (99.5th percentile and percentile ranks).

Each procedure has its own full run: search, selection on 2024 and final models on 2025.
Both are evaluated on the same 2025 students with the same bootstrap samples, so every
comparison is paired. Table 8 and figure 5.

    python -m src.grade_harmonization
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

import config
from src import figures
from src.reproducibility import setup_logging
from src.statistics import bootstrap_p_value

log = logging.getLogger("harmonization")
LABELS = {"p995": "Grades rescaled to 0-10 (99.5th percentile)",
          "rank": "Grades as percentile ranks (search repeated)"}


def load(folder):
    tables = Path(folder) / "tables"
    main = pd.read_csv(tables / "main_results.csv", index_col=0).loc[config.MODELS]
    combos = pd.read_csv(tables / "metrics_ensemble_all_combinations.csv").set_index(["model", "strategy"])
    boot = pd.read_csv(tables / "bootstrap_samples.csv")
    boot = boot[boot["metric"] == "ap"].reset_index(drop=True)[config.MODELS]
    return main, combos, boot


def compare(p995_dir, rank_dir):
    main_a, combos_a, boot_a = load(p995_dir)
    main_b, combos_b, boot_b = load(rank_dir)

    rows = []
    for m in config.MODELS:
        d = boot_b[m] - boot_a[m]
        rows.append({"model": m, "strategy_p995": main_a.loc[m, "strategy"], "ap_p995": main_a.loc[m, "ap"],
                     "ap_p995_low": main_a.loc[m, "ap_low"], "ap_p995_high": main_a.loc[m, "ap_high"],
                     "auc_p995": main_a.loc[m, "auc"], "strategy_rank": main_b.loc[m, "strategy"],
                     "ap_rank": main_b.loc[m, "ap"], "ap_rank_low": main_b.loc[m, "ap_low"],
                     "ap_rank_high": main_b.loc[m, "ap_high"], "auc_rank": main_b.loc[m, "auc"],
                     "recall_top5_rank": main_b.loc[m, "recall_top5"],
                     "diff": main_b.loc[m, "ap"] - main_a.loc[m, "ap"],
                     "diff_low": d.quantile(0.025), "diff_high": d.quantile(0.975)})
    table = pd.DataFrame(rows)

    gap = (boot_a["RF"] - boot_a["LR"]) - (boot_b["RF"] - boot_b["LR"])
    joint = combos_a.join(combos_b, rsuffix="_rank")
    first = {name: b.idxmax(axis=1).value_counts(normalize=True).reindex(config.MODELS, fill_value=0)
             for name, b in (("p995", boot_a), ("rank", boot_b))}
    summary = {
        "share_rf_above_lr_p995": float((boot_a["RF"] > boot_a["LR"]).mean()),
        "share_lr_above_rf_rank": float((boot_b["LR"] > boot_b["RF"]).mean()),
        "rf_minus_lr_gap_change": float((main_a.loc["RF", "ap"] - main_a.loc["LR", "ap"])
                                        - (main_b.loc["RF", "ap"] - main_b.loc["LR", "ap"])),
        "gap_change_low": float(gap.quantile(0.025)), "gap_change_high": float(gap.quantile(0.975)),
        "gap_change_p": bootstrap_p_value(gap),
        "kendall_tau_model_order": float(kendalltau(main_a["ap"], main_b["ap"]).statistic),
        "spearman_27_combinations": float(spearmanr(joint["ap"], joint["ap_rank"]).statistic),
        "validation_higher_with_ranks": int((joint["val_ap_rank"] > joint["val_ap"]).sum()),
        "test_higher_with_ranks": int((joint["ap_rank"] > joint["ap"]).sum()),
        "first_place_p995": first["p995"].to_dict(), "first_place_rank": first["rank"].to_dict(),
    }
    return table, summary, {"p995": main_a, "rank": main_b}, first


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--rank-dir", default=str(config.OUTPUT_DIR / "grade_rank"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    out = Path(args.output_dir)
    setup_logging(out / "logs", args.log_level, "harmonization")

    table, summary, results, first = compare(out, args.rank_dir)
    table.to_csv(out / "tables" / "grade_harmonization.csv", index=False)
    (out / "tables" / "grade_harmonization_summary.json").write_text(json.dumps(summary, indent=1))
    figures.plot_grade_harmonization(results, first, LABELS, out / "figures")
    log.info("RF above LR with the 99.5th percentile: %.3f; LR above RF with ranks: %.3f",
             summary["share_rf_above_lr_p995"], summary["share_lr_above_rf_rank"])


if __name__ == "__main__":
    main()
