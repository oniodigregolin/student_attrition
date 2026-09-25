"""Classifiers, imbalance strategies and hyperparameter draws."""
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config
from src.reproducibility import make_seed


def combinations():
    """The 27 model/strategy pairs of the study."""
    pairs = [(m, s) for m in config.GENERAL_MODELS for s in config.STRATEGIES]
    return pairs + [(m, "built-in") for m in config.BUILTIN_MODELS]


def _pick(rng, options):
    return options[rng.integers(len(options))]


def _draw(rng, space):
    params = {}
    for name, options in space.items():
        if isinstance(options, tuple) and options[0] == "loguniform":
            params[name] = float(loguniform(options[1], options[2]).rvs(random_state=rng))
        else:
            params[name] = _pick(rng, options)
    return params


def draw_setting(model, strategy, index, base_seed=config.BASE_SEED):
    """Hyperparameter setting number ``index`` of the random search for (model, strategy)."""
    rng = np.random.default_rng(make_seed(model, strategy, index, "search_draw", base_seed=base_seed))
    params = _draw(rng, config.SEARCH_SPACES[model])
    if strategy == "weight":
        params |= _draw(rng, config.STRATEGY_SPACES["weight"])
    if strategy in ("RUS", "ROS", "SMOTE"):
        params |= _draw(rng, config.STRATEGY_SPACES["resampling"])
    if strategy == "SMOTE":
        params |= _draw(rng, config.STRATEGY_SPACES["smote"])
    return params


def positive_weight(params, ratio):
    """Weight of the positive class; ``ratio`` = negatives / positives in the training data."""
    if "w_mode" not in params:
        return 1.0
    if params["w_mode"] == "sqrt":
        return float(np.sqrt(ratio))
    return float(ratio * params["w_mult"])


def make_classifier(model, p, seed, w=1.0):
    weights = {0: 1, 1: w}
    if model == "LR":
        return LogisticRegression(C=p["C"], class_weight=weights, max_iter=3000)
    if model == "DT":
        return DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["msl"],
                                      class_weight=weights, random_state=seed)
    if model == "SVM":
        # RBF kernel approximated with Nystroem, then a linear SVM in that space;
        # an exact kernel SVM is not feasible for ~47k training rows
        return Pipeline([("nystroem", Nystroem(gamma=p["gamma"], n_components=config.NYSTROEM_COMPONENTS,
                                               random_state=seed)),
                         ("svc", LinearSVC(C=p["C"], class_weight=weights, max_iter=5000))])
    if model == "RF":
        return RandomForestClassifier(n_estimators=config.N_TREES, max_depth=p["max_depth"],
                                      min_samples_leaf=p["msl"], max_features=p["mf"],
                                      class_weight=weights, random_state=seed, n_jobs=1)
    if model == "XGB":
        return XGBClassifier(n_estimators=p["n"], max_depth=p["depth"], learning_rate=p["lr"],
                             min_child_weight=p["mcw"], subsample=p["ss"], colsample_bytree=p["cs"],
                             scale_pos_weight=w, tree_method="hist", random_state=seed,
                             n_jobs=1, verbosity=0)
    if model == "BRF":
        return BalancedRandomForestClassifier(n_estimators=config.N_TREES, max_depth=p["max_depth"],
                                              min_samples_leaf=p["msl"], max_features=p["mf"],
                                              sampling_strategy=p["ratio"], replacement=True,
                                              bootstrap=False, random_state=seed, n_jobs=1)
    if model == "RUSBoost":
        return RUSBoostClassifier(estimator=DecisionTreeClassifier(max_depth=p["depth"]),
                                  n_estimators=p["n"], learning_rate=p["lr"],
                                  sampling_strategy=p["ratio"], random_state=seed)
    raise ValueError(f"unknown model {model}")


def make_pipeline(model, strategy, p, seed, ratio):
    """Imputation and scaling, resampling (training data only) and the classifier."""
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if strategy == "RUS":
        steps.append(("resample", RandomUnderSampler(sampling_strategy=p["ratio_s"], random_state=seed)))
    elif strategy == "ROS":
        steps.append(("resample", RandomOverSampler(sampling_strategy=p["ratio_s"], random_state=seed)))
    elif strategy == "SMOTE":
        steps.append(("resample", SMOTE(sampling_strategy=p["ratio_s"], k_neighbors=p["k"], random_state=seed)))
    steps.append(("model", make_classifier(model, p, seed, positive_weight(p, ratio))))
    return Pipeline(steps)


def fit_and_score(model, strategy, params, seed, X_train, y_train, X_test):
    """Fit one pipeline and return scores for X_test (decision function for the SVM)."""
    if len(np.unique(y_train)) < 2:
        raise ValueError("training data contain a single class")
    ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    pipe = make_pipeline(model, strategy, params, seed, ratio)
    pipe.fit(X_train, y_train)
    if model == "SVM":
        return pipe.decision_function(X_test)
    return pipe.predict_proba(X_test)[:, 1]


def has_probabilities(model):
    return model != "SVM"
