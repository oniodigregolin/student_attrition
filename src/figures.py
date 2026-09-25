"""Figures, drawn with the libraries' defaults and saved as SVG and PNG."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scikit_posthocs as sp
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

LABELS = {"LR": "Logistic regression", "DT": "Decision tree", "SVM": "SVM (Nystroem)",
          "RF": "Random forest", "XGB": "XGBoost", "BRF": "Balanced random forest",
          "RUSBoost": "RUSBoost", "Baseline": "Additive score (no learning)"}
STRATEGY_LABELS = {"none": "None", "weight": "Class weight", "RUS": "RUS", "ROS": "ROS",
                   "SMOTE": "SMOTE", "built-in": "Built-in"}


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.svg")
    fig.savefig(folder / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_performance_intervals(main, chosen, prevalence, folder):
    order = main.sort_values("ap").index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (col, title) in zip(axes, [("ap", "PR-AUC"), ("auc", "ROC-AUC"),
                                       ("recall_top5", "Recall in the top-5% list")]):
        est = main.loc[order, col]
        err = [est - main.loc[order, col + "_low"], main.loc[order, col + "_high"] - est]
        ax.errorbar(est, range(len(order)), xerr=err, fmt="o", capsize=3)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].axvline(prevalence, linestyle="--", color="gray", label="Prevalence")
    axes[0].legend(loc="lower right")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{LABELS[m]} ({STRATEGY_LABELS[chosen[m]]})" for m in order])
    fig.tight_layout()
    save(fig, folder, "fig2_performance")


def plot_confusion_matrices(y, flagged, models, folder):
    """Cells are shaded by the share of the true class (row), so the rare class stays readable;
    each cell shows the count and that share."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, m in zip(axes.ravel(), models):
        counts = confusion_matrix(y, flagged[m], labels=[0, 1])
        shares = counts / counts.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(shares, display_labels=["Stayed", "Left"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".1%")
        disp.im_.set_clim(0, 1)
        for (i, j), text in np.ndenumerate(disp.text_):
            r, g, b, _ = disp.im_.cmap(disp.im_.norm(shares[i, j]))
            text.set_text(f"{counts[i, j]:,}\n({shares[i, j]:.1%})")
            text.set_color("black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white")
        ax.set_title(LABELS[m])
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, folder, "fig4_confusion_matrices")


def plot_strategy_heatmaps(ap_table, prob_table, folder):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, table, fmt, title in [(axes[0], ap_table, ".3f", "PR-AUC (2025 test set)"),
                                  (axes[1], prob_table, ".1f", "Mean predicted probability (%)")]:
        t = table.rename(index=LABELS, columns=STRATEGY_LABELS)
        sns.heatmap(t, annot=True, fmt=fmt, ax=ax)
        ax.set_title(title)
    fig.tight_layout()
    save(fig, folder, "fig6_strategies")


def plot_rarity_experiment(summary, models, prevalence, folder):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        part = summary[summary["model"] == m].sort_values("n_pos")
        ax.plot(part["n_pos"], part["ap_mean"], marker="o", label=LABELS[m])
    ax.axhline(prevalence, linestyle="--", color="gray", label="Prevalence")
    ax.set_xscale("log")
    ticks = summary.groupby("level")["n_pos"].mean().sort_values()
    ax.set_xticks(ticks.values)
    ax.set_xticklabels([str(int(t)) for t in ticks.values])
    ax.minorticks_off()
    ax.set_xlabel("Departures kept in the training data")
    ax.set_ylabel("PR-AUC on the 2025 test set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, folder, "fig7_rarity")


def plot_rank_heatmap(ranks, folder):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(ranks.rename(index=LABELS), annot=True, fmt="d", cmap="viridis_r", ax=ax)
    ax.set_title("Rank of each model by metric and alert-list size (1 = best)")
    fig.tight_layout()
    save(fig, folder, "fig3_ranks_by_metric")


def plot_school_cd(mean_ranks, nemenyi, n_schools, folder):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sp.critical_difference_diagram(mean_ranks.rename(LABELS), nemenyi.rename(index=LABELS, columns=LABELS), ax=ax)
    ax.set_title(f"Mean rank of PR-AUC within schools ({n_schools} schools)")
    fig.tight_layout()
    save(fig, folder, "fig8_school_cd")


def plot_overlap(jaccard, folder):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(jaccard.rename(index=LABELS, columns=LABELS), annot=True, fmt=".2f", ax=ax)
    ax.set_title("Jaccard index between top-5% lists")
    fig.tight_layout()
    save(fig, folder, "fig9_overlap")


def plot_grade_harmonization(results, first_place, labels, folder):
    """PR-AUC with 95% intervals under each grade harmonization, and first-place shares.

    ``results`` and ``first_place`` map a procedure label to main_results and to the
    share of bootstrap samples in which each model had the highest PR-AUC.
    """
    names = list(results)
    order = results[names[0]].sort_values("ap").index.tolist()
    y = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    for (name, df), off in zip(results.items(), (0.15, -0.15)):
        est = df.loc[order, "ap"]
        err = [est - df.loc[order, "ap_low"], df.loc[order, "ap_high"] - est]
        ax1.errorbar(est, y + off, xerr=err, fmt="o", capsize=3, label=labels[name])
        ax2.barh(y + off + (0.05 if off > 0 else -0.05), first_place[name].reindex(order), height=0.4, label=labels[name])
    ax1.set_yticks(y)
    ax1.set_yticklabels([LABELS[m] for m in order])
    ax1.set_xlabel("PR-AUC on the 2025 test set (95% bootstrap interval)")
    ax2.set_xlabel("Share of bootstrap samples with the highest PR-AUC")
    ax2.set_xlim(0, 1)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    handles, names = ax1.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, folder, "fig5_grade_harmonization")
