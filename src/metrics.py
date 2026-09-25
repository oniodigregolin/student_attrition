"""Metrics for ranked alert lists and probabilities.

Top-k lists: when several students share the score at the cut-off of the list, the
reported numbers are the *expected* hits under random tie-breaking, so they do not depend
on any random generator. Only the decision tree has ties large enough to matter.
A concrete list (for confusion-matrix figures or overlaps) uses a local seed.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

from src.reproducibility import make_seed


def list_size(n, share):
    return max(1, int(round(share * n)))


def expected_hits(y, scores, k):
    """Expected number of positives in the top-k list, ties broken at random."""
    y, scores = np.asarray(y), np.asarray(scores)
    cutoff = np.partition(scores, -k)[-k]
    above = scores > cutoff
    tied = scores == cutoff
    return y[above].sum() + (k - above.sum()) * y[tied].mean()


def top_k_indices(scores, k, seed):
    """One concrete top-k list; ties at the cut-off are broken with a local generator."""
    noise = np.random.default_rng(seed).random(len(scores))
    return np.lexsort((noise, -np.asarray(scores)))[:k]


def recall_at_k(y, scores, k):
    return expected_hits(y, scores, k) / np.sum(y)


def precision_at_k(y, scores, k):
    return expected_hits(y, scores, k) / k


def calculate_model_metrics(y, scores, has_prob=True, prevalence_ref=None):
    """PR-AUC, ROC-AUC, list metrics and, for probabilities, Brier and Brier skill score.

    The Brier skill score uses as reference a constant forecast equal to
    ``prevalence_ref`` (the training prevalence, known when the model is used).
    """
    n = len(y)
    out = {"ap": average_precision_score(y, scores), "auc": roc_auc_score(y, scores)}
    for share, label in [(0.01, "1"), (0.05, "5")]:
        k = list_size(n, share)
        out[f"recall_top{label}"] = recall_at_k(y, scores, k)
        out[f"precision_top{label}"] = precision_at_k(y, scores, k)
    if has_prob:
        out["brier"] = brier_score_loss(y, scores)
        out["mean_prob"] = float(np.mean(scores))
        if prevalence_ref is not None:
            ref = brier_score_loss(y, np.full(n, prevalence_ref))
            out["brier_skill"] = 1 - out["brier"] / ref
    return out


def calculate_default_threshold_metrics(y, scores, has_prob=True):
    """Confusion matrix at the libraries' default threshold (0.5, or 0 for the SVM)."""
    flagged = (np.asarray(scores) >= (0.5 if has_prob else 0.0)).astype(int)
    return calculate_confusion_matrix_metrics(y, flagged)


def calculate_confusion_matrix_metrics(y, flagged):
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    return {"flagged": int(tp + fp), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "recall": tp / max(tp + fn, 1), "precision": tp / (tp + fp) if tp + fp else np.nan,
            "specificity": tn / max(tn + fp, 1)}


def top_k_confusion(y, scores, share, model, strategy):
    """Confusion matrix of one concrete top-k list (seeded tie-breaking)."""
    k = list_size(len(y), share)
    flagged = np.zeros(len(y), dtype=int)
    flagged[top_k_indices(scores, k, make_seed(model, strategy, int(share * 1000), "tie"))] = 1
    return calculate_confusion_matrix_metrics(y, flagged), flagged


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calculate_calibration_metrics(y, p):
    """Calibration-in-the-large, observed/expected ratio, calibration intercept and slope.

    The intercept is estimated with the slope fixed at 1 (logit(p) as offset); the slope
    comes from a logistic regression of the outcome on logit(p).
    """
    y = np.asarray(y)
    lp = _logit(p)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0][0]
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp)))
        step = (y - q).sum() / (q * (1 - q)).sum()
        a += step
        if abs(step) < 1e-10:
            break
    return {"observed": y.mean(), "mean_pred": float(np.mean(p)),
            "citl": y.mean() - float(np.mean(p)), "o_e": y.mean() / float(np.mean(p)),
            "intercept": a, "slope": slope}
