"""Run the whole study, or some of its steps, in order.

    python -m scripts.run_pipeline
    python -m scripts.run_pipeline --steps results extra harmonization

Each step is also a module that can be run on its own (see README).
"""
import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config
from src.reproducibility import setup_logging

log = logging.getLogger("pipeline")
STEPS = ["prepare", "fit", "rarity", "results", "extra", "grade_rank", "harmonization"]


def commands(args):
    out = ["--output-dir", args.output_dir, "--log-level", args.log_level]
    rank = ["--output-dir", str(Path(args.output_dir) / "grade_rank"), "--log-level", args.log_level]
    jobs = ["--n-jobs", str(args.n_jobs)]
    force = ["--force"] if args.force else []
    return {
        "prepare": [["src.prepare_data", "--data-file", args.data_file] + out,
                    ["src.prepare_data", "--data-file", args.data_file, "--grade-method", "rank"] + out],
        "fit": [["src.fit_models"] + jobs + force + out],
        "rarity": [["src.rarity_experiment"] + jobs + force + out],
        "results": [["src.build_results"] + out],
        "extra": [["src.extra_analyses", "--robustness"] + jobs + out],
        "grade_rank": [["src.fit_models"] + jobs + force + rank, ["src.build_results"] + rank],
        "harmonization": [["src.grade_harmonization"] + out],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    parser.add_argument("--data-file", default=str(config.DATA_FILE))
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--force", action="store_true", help="refit even when cached results match")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(Path(args.output_dir) / "logs", args.log_level, "pipeline")

    cmds = commands(args)
    for step in [s for s in STEPS if s in args.steps]:
        log.info("Step %s started", step)
        start = time.time()
        if step == "grade_rank":
            # the same protocol, run separately on the grades converted to percentile ranks
            rank_dir = Path(args.output_dir) / "grade_rank"
            rank_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(args.output_dir) / "analytic_grade-rank.pkl", rank_dir / "analytic.pkl")
        for cmd in cmds[step]:
            result = subprocess.run([sys.executable, "-m"] + cmd)
            if result.returncode != 0:
                log.error("Step %s failed (exit code %d); later steps were not run", step, result.returncode)
                sys.exit(result.returncode)
        log.info("Step %s finished in %.0f s", step, time.time() - start)


if __name__ == "__main__":
    main()
