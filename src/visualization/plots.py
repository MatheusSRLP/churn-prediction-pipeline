import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)

_SOURCE = "Source: IBM Telco Customer Churn Dataset"

# ---------------------------------------------------------------------------
# Global style — applied once at import time
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
sns.set_palette("colorblind")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_source(fig: plt.Figure) -> None:
    """Add a data-source footnote to the bottom-left of the figure."""
    fig.text(0.01, -0.02, _SOURCE, ha="left", va="top", fontsize=8, color="gray")


def _save(fig: plt.Figure, path: Path, output_dir: Path) -> Path:
    """Save figure, log path, close to free memory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / path
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved to %s", full_path)
    return full_path.resolve()


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_churn_distribution(
    df: pd.DataFrame,
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """Bar chart showing absolute counts and percentage labels for each churn class.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame containing a numeric 'Churn' column (0/1).
    output_dir : Path
        Directory where the figure is saved.

    Returns
    -------
    Path
        Resolved path to the saved figure.
    """
    counts = df["Churn"].value_counts().sort_index()
    labels = ["No Churn (0)", "Churn (1)"]
    total = counts.sum()

    fig, ax = plt.subplots()
    bars = ax.bar(labels, counts.values, color=sns.color_palette("colorblind")[:2], width=0.5)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _add_source(fig)
    return _save(fig, "churn_distribution.png", output_dir)


def plot_churn_by_contract(
    df: pd.DataFrame,
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """Grouped bar chart of churn rate (%) by contract type.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'Contract' and 'Churn' (0/1) columns.
    output_dir : Path

    Returns
    -------
    Path
    """
    summary = (
        df.groupby("Contract")["Churn"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "churned", "count": "total"})
    )
    summary["churn_rate"] = summary["churned"] / summary["total"] * 100
    summary = summary.sort_values("churn_rate", ascending=False).reset_index()

    palette = sns.color_palette("colorblind", n_colors=len(summary))
    fig, ax = plt.subplots()
    bars = ax.bar(
        summary["Contract"],
        summary["churn_rate"],
        color=palette,
        width=0.5,
    )

    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row.churn_rate:.1f}%\n(n={row.total:,})",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Contract Type")
    ax.set_ylim(0, summary["churn_rate"].max() * 1.25)
    _add_source(fig)
    return _save(fig, "churn_by_contract.png", output_dir)


def plot_tenure_distribution(
    df: pd.DataFrame,
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """Overlapping histogram of tenure in months, split by churn label.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'tenure' (int) and 'Churn' (0/1) columns.
    output_dir : Path

    Returns
    -------
    Path
    """
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots()

    for label, color, name in [
        (0, palette[0], "No Churn"),
        (1, palette[1], "Churn"),
    ]:
        ax.hist(
            df.loc[df["Churn"] == label, "tenure"],
            bins=30,
            alpha=0.6,
            color=color,
            label=name,
            edgecolor="none",
        )

    ax.set_title("Tenure Distribution by Churn Label", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Number of Customers")
    ax.legend(title="Churn", frameon=False)
    _add_source(fig)
    return _save(fig, "tenure_distribution.png", output_dir)


def plot_charges_boxplot(
    df: pd.DataFrame,
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """Side-by-side boxplot of MonthlyCharges for churned vs. retained customers.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'MonthlyCharges' and 'Churn' (0/1) columns.
    output_dir : Path

    Returns
    -------
    Path
    """
    plot_df = df[["MonthlyCharges", "Churn"]].copy()
    plot_df["Churn Label"] = plot_df["Churn"].map({0: "No Churn", 1: "Churn"})

    fig, ax = plt.subplots()
    sns.boxplot(
        data=plot_df,
        x="Churn Label",
        y="MonthlyCharges",
        hue="Churn Label",
        palette="colorblind",
        width=0.4,
        linewidth=1.2,
        legend=False,
        ax=ax,
    )

    ax.set_title("Monthly Charges Distribution by Churn Label", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Monthly Charges (USD)")
    _add_source(fig)
    return _save(fig, "charges_boxplot.png", output_dir)


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models_proba: dict[str, np.ndarray],
    y_test: pd.Series,
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """Plot ROC curves for multiple models on a single set of axes.

    Parameters
    ----------
    models_proba : dict[str, np.ndarray]
        Mapping of model_name → 1-D array of predicted positive probabilities.
    y_test : pd.Series
        Ground-truth binary labels (0/1).
    output_dir : Path

    Returns
    -------
    Path
    """
    palette = sns.color_palette("colorblind", n_colors=len(models_proba))
    fig, ax = plt.subplots()

    for (name, proba), color in zip(models_proba.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    _add_source(fig)
    return _save(fig, "roc_curves.png", output_dir)


def plot_pr_curves(
    models_proba: dict[str, np.ndarray],
    y_test: pd.Series,
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """Plot Precision-Recall curves for multiple models on a single set of axes.

    PR-AUC is more informative than ROC-AUC for imbalanced classes because it
    does not count true negatives — avoiding the ROC optimism trap.

    Parameters
    ----------
    models_proba : dict[str, np.ndarray]
        Mapping of model_name → 1-D array of predicted positive probabilities.
    y_test : pd.Series
        Ground-truth binary labels (0/1).
    output_dir : Path

    Returns
    -------
    Path
    """
    palette = sns.color_palette("colorblind", n_colors=len(models_proba))
    baseline = y_test.mean()

    fig, ax = plt.subplots()

    for (name, proba), color in zip(models_proba.items(), palette):
        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP = {pr_auc:.4f})")

    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Random classifier ({baseline:.2f})")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    _add_source(fig)
    return _save(fig, "pr_curves.png", output_dir)


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """Normalised (by true label) confusion matrix with annotation.

    Parameters
    ----------
    y_test : pd.Series
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels at the chosen decision threshold.
    model_name : str
        Used in the plot title and filename.
    output_dir : Path

    Returns
    -------
    Path
    """
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    display_labels = ["No Churn (0)", "Churn (1)"]

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    _add_source(fig)

    safe_name = model_name.lower().replace(" ", "_")
    return _save(fig, f"confusion_matrix_{safe_name}.png", output_dir)


def plot_calibration_curves(
    models_proba: dict[str, np.ndarray],
    y_test: pd.Series,
    output_dir: Path = Path("outputs/figures"),
    n_bins: int = 10,
) -> Path:
    """Reliability diagram: predicted probability vs. observed fraction of positives.

    A well-calibrated model lies on the diagonal. Points above the diagonal
    indicate under-confidence; below indicates over-confidence.

    Parameters
    ----------
    models_proba : dict[str, np.ndarray]
        Mapping of model_name → predicted positive probabilities.
    y_test : pd.Series
        Ground-truth binary labels (0/1).
    output_dir : Path
    n_bins : int
        Number of probability bins (default 10).

    Returns
    -------
    Path
    """
    palette = sns.color_palette("colorblind", n_colors=len(models_proba))

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for (name, proba), color in zip(models_proba.items(), palette):
        fraction_pos, mean_pred = calibration_curve(
            y_test, proba, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_pred, fraction_pos, "o-", color=color, lw=2, label=name)

    ax.set_title("Calibration Curves (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    _add_source(fig)
    return _save(fig, "calibration_curves.png", output_dir)


# ---------------------------------------------------------------------------
# Interpretability plots
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    output_dir: Path = Path("outputs/figures"),
) -> Path:
    """SHAP beeswarm summary plot (global feature importance).

    Each dot is one prediction. Position on the x-axis shows the SHAP value
    (impact on model output); colour shows feature value (red = high, blue = low).

    Parameters
    ----------
    shap_values : np.ndarray
        2-D array of shape (n_samples, n_features) from shap.Explainer.
    X_test : pd.DataFrame
        Feature matrix corresponding to shap_values rows — provides feature names.
    output_dir : Path

    Returns
    -------
    Path
    """
    import shap  # optional import — keeps shap out of the module-level namespace

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
        plot_type="dot",
    )
    fig = plt.gcf()
    fig.suptitle("SHAP Summary — Global Feature Importance", fontsize=14, fontweight="bold", y=1.01)
    _add_source(fig)
    return _save(fig, "shap_summary.png", output_dir)
