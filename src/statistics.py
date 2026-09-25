"""Bootstrap comparisons and the other tests used in the study.

The bootstrap resamples the 2025 test set with the predictions held fixed, so the intervals
reflect uncertainty due to the test sample, not the variability of training.
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr


def stratified_bootstrap_indices(y, n_boot, seed):
    """Yield index arrays that resample positives and negatives separately.

    Class sizes and the test size are kept, and the same indices are used for every model
    so that comparisons are paired.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(np.asarray(y) == 1), np.flatnonzero(np.asarray(y) == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_p_value(diffs):
    """Two-sided p-value for a paired bootstrap difference.

    Uses (count + 1) / (N + 1) on each side, so the smallest possible value is
    2 / (N + 1) and p = 0 is never reported.
    """
    diffs = np.asarray(diffs)
    n = len(diffs)
    below = ((diffs <= 0).sum() + 1) / (n + 1)
    above = ((diffs >= 0).sum() + 1) / (n + 1)
    return min(1.0, 2 * min(below, above))


def paired_bootstrap_comparison(values, reference):
    """Differences of each model against ``reference`` from a dict of bootstrap values."""
    rows = []
    for name, v in values.items():
        if name == reference:
            continue
        d = np.asarray(v) - np.asarray(values[reference])
        rows.append({"model": name, "diff_low": np.quantile(d, 0.025), "diff_high": np.quantile(d, 0.975),
                     "p_raw": bootstrap_p_value(d)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_raw"].to_numpy())
    return table


def holm_adjust(p):
    """Holm step-down adjustment (same as statsmodels multipletests(method='holm'))."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def friedman_strategy_test(table):
    """Friedman test over the columns of ``table`` (rows = blocks).

    With algorithms as blocks the blocks are not independent samples of a population of
    datasets, so the result is exploratory.
    """
    stat, p = friedmanchisquare(*[table[c] for c in table.columns])
    return {"blocks": list(table.index), "treatments": list(table.columns), "chi2": stat, "p": p,
            "values": table.to_dict(), "mean_rank": table.rank(axis=1, ascending=False).mean().to_dict()}


def validation_test_correlation(val, test):
    rho, p = spearmanr(val, test)
    return {"spearman": rho, "p": p, "n": len(val)}


def bootstrap_winner_frequency(matrix, names):
    """Proportion of bootstrap samples in which each model had the highest value.

    ``matrix`` has one row per bootstrap sample and one column per model. Ties are split
    equally between the tied models. This is a frequency, not a Bayesian probability.
    """
    matrix = np.asarray(matrix)
    wins = np.zeros(matrix.shape[1])
    for row in matrix:
        best = np.flatnonzero(row == row.max())
        wins[best] += 1 / len(best)
    return pd.Series(wins / len(matrix), index=names)
