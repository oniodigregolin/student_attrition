"""Seeds, hashes, run manifests and logging."""
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
import warnings
from pathlib import Path

import config

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost",
            "matplotlib", "seaborn", "joblib", "openpyxl", "scikit-posthocs"]


def make_seed(model, strategy, setting_index=0, stage="final_fit", repetition=0,
              base_seed=config.BASE_SEED):
    """Deterministic seed for each random step of the study.

    Stages:
    - search_draw:   hyperparameter draw ``setting_index`` for (model, strategy)
    - search_fit:    classifier/resampler seed during the search (always the base seed)
    - final_fit:     final models, one per ``repetition`` (seeds 0-4)
    - rarity_subset: positive subset; ``setting_index`` holds the level in percent.
                     It does not depend on the model, so every model sees the same subset.
    - rarity_fit:    model seed in the rarity experiment
    - tie:           random tie-breaking in top-k lists
    """
    if stage == "search_draw":
        return base_seed + 1000 * setting_index + sum(map(ord, model + strategy)) % 997
    if stage == "search_fit":
        return base_seed
    if stage in ("final_fit", "rarity_fit"):
        return base_seed + repetition
    if stage == "rarity_subset":
        return base_seed + 100 * repetition + setting_index
    if stage == "tie":
        text = f"{model}|{strategy}|{setting_index}|{repetition}"
        return base_seed + int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    raise ValueError(f"unknown seed stage: {stage}")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(obj):
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def library_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def study_settings():
    """The parts of config.py that define a fitted model. Used to validate the cache."""
    return {"features": config.FEATURES, "search_spaces": config.SEARCH_SPACES,
            "strategy_spaces": config.STRATEGY_SPACES, "n_settings": config.N_SETTINGS,
            "seeds": config.SEEDS, "base_seed": config.BASE_SEED, "n_trees": config.N_TREES,
            "nystroem": config.NYSTROEM_COMPONENTS, "train_year": config.TRAIN_YEAR,
            "validation_year": config.VALIDATION_YEAR,
            "final_train_years": config.FINAL_TRAIN_YEARS, "test_year": config.TEST_YEAR}


def make_manifest(analytic_path, **extra):
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "libraries": library_versions(),
        "analytic_hash": file_hash(analytic_path),
        "config_hash": object_hash(study_settings()),
    }
    manifest.update(extra)
    return manifest


def manifests_match(old, new, keys=("analytic_hash", "config_hash", "model", "strategy",
                                    "features", "n_settings", "seeds", "base_seed")):
    """Return the keys that differ between two manifests (empty list if compatible)."""
    return [k for k in keys if old.get(k) != new.get(k)]


def setup_logging(log_dir, level="INFO", name="run"):
    """Log to the console and to outputs/logs; warnings go to model_warnings.log."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    capture_warnings(log_dir)


def worker_logging(log_dir, name, level="INFO"):
    """Make a joblib worker (a separate process) write to the same log file as the main process."""
    target = os.path.abspath(Path(log_dir) / f"{name}.log")
    root = logging.getLogger()
    if not any(getattr(h, "baseFilename", None) == target for h in root.handlers):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(level)
    capture_warnings(log_dir)


def capture_warnings(log_dir):
    """Send Python warnings to outputs/logs/model_warnings.log instead of hiding them.

    Called again inside joblib workers, which run in separate processes.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.captureWarnings(True)
    wlog = logging.getLogger("py.warnings")
    target = os.path.abspath(Path(log_dir) / "model_warnings.log")
    if not any(getattr(h, "baseFilename", None) == target for h in wlog.handlers):
        handler = logging.FileHandler(target)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        wlog.addHandler(handler)
        wlog.propagate = False
    # repeated identical warnings are logged once per location
    warnings.simplefilter("default")
