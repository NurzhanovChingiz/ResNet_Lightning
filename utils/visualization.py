"""
Visualization utilities for ResNet Lightning training logs.

Reads the CSV metrics logged by Lightning's CSVLogger and produces
publication-quality plots saved alongside the CSV file.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path


# ── Colour palette ──────────────────────────────────────────────────────────
TRAIN_COLOR = "#2196F3"   # blue
VAL_COLOR   = "#FF9800"   # orange
TEST_COLOR  = "#4CAF50"   # green
GRID_ALPHA  = 0.25


def _find_latest_metrics_csv(log_dir: str = "lightning_logs") -> str:
    """Return the path to the metrics.csv inside the highest-version folder."""
    versions = sorted(
        glob.glob(os.path.join(log_dir, "version_*")),
        key=lambda p: int(p.split("_")[-1]),
    )
    if not versions:
        raise FileNotFoundError(f"No version folders found in '{log_dir}'")
    csv_path = os.path.join(versions[-1], "metrics.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metrics.csv not found in '{versions[-1]}'")
    return csv_path


def load_metrics(csv_path: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load and split the Lightning CSV into per-split DataFrames indexed by epoch.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a DataFrame indexed by epoch.
    """
    if csv_path is None:
        csv_path = _find_latest_metrics_csv()

    df = pd.read_csv(csv_path)

    # ── Train rows: have train_loss, no val_loss ────────────────────────
    train_cols = [c for c in df.columns if c.startswith("train_")]
    train_df = df.loc[df["train_loss"].notna(), ["epoch", "step"] + train_cols].copy()
    train_df = train_df.set_index("epoch").drop(columns="step")
    # Remove the 'train_' prefix for uniform naming
    train_df.columns = [c.replace("train_", "") for c in train_df.columns]

    # ── Val rows: have val_loss, no train_loss ──────────────────────────
    val_cols = [c for c in df.columns if c.startswith("val_")]
    val_df = df.loc[df["val_loss"].notna(), ["epoch", "step"] + val_cols].copy()
    val_df = val_df.set_index("epoch").drop(columns="step")
    val_df.columns = [c.replace("val_", "") for c in val_df.columns]

    # ── Test rows: have test_loss ───────────────────────────────────────
    test_cols = [c for c in df.columns if c.startswith("test_")]
    test_df = df.loc[df["test_loss"].notna(), ["epoch", "step"] + test_cols].copy()
    if not test_df.empty:
        test_df = test_df.set_index("epoch").drop(columns="step")
        test_df.columns = [c.replace("test_", "") for c in test_df.columns]

    return {"train": train_df, "val": val_df, "test": test_df}


# ── Individual plotters ─────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "Epoch"):
    """Apply consistent styling to an Axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(labelsize=9)


def plot_loss(
    metrics: dict[str, pd.DataFrame],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot train & val loss curves (with test point if available)."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    train, val, test = metrics["train"], metrics["val"], metrics["test"]

    ax.plot(train.index, train["loss"], color=TRAIN_COLOR, label="Train Loss", linewidth=1.8, marker="o", markersize=3)
    ax.plot(val.index, val["loss"], color=VAL_COLOR, label="Val Loss", linewidth=1.8, marker="s", markersize=3)

    if not test.empty and "loss" in test.columns:
        # Show test as a horizontal dashed line spanning the full x range
        test_loss = test["loss"].iloc[0]
        ax.axhline(y=test_loss, color=TEST_COLOR, linestyle="--", linewidth=1.5, label=f"Test Loss ({test_loss:.4f})")

    _style_ax(ax, "Loss vs. Epoch", "Loss")

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_accuracy(
    metrics: dict[str, pd.DataFrame],
    ax_top1: plt.Axes | None = None,
    ax_top5: plt.Axes | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot Top-1 and Top-5 accuracy curves."""
    show = ax_top1 is None
    if ax_top1 is None or ax_top5 is None:
        fig, (ax_top1, ax_top5) = plt.subplots(1, 2, figsize=(14, 5))

    train, val, test = metrics["train"], metrics["val"], metrics["test"]

    # ── Top-1 ───────────────────────────────────────────────────────────
    ax_top1.plot(train.index, train["acc_top1"] * 100, color=TRAIN_COLOR, label="Train", linewidth=1.8, marker="o", markersize=3)
    ax_top1.plot(val.index, val["acc_top1"] * 100, color=VAL_COLOR, label="Val", linewidth=1.8, marker="s", markersize=3)
    if not test.empty and "acc_top1" in test.columns:
        val_acc = test["acc_top1"].iloc[0] * 100
        ax_top1.axhline(y=val_acc, color=TEST_COLOR, linestyle="--", linewidth=1.5, label=f"Test ({val_acc:.2f}%)")
    _style_ax(ax_top1, "Top-1 Accuracy", "Accuracy (%)")

    # ── Top-5 ───────────────────────────────────────────────────────────
    ax_top5.plot(train.index, train["acc_top5"] * 100, color=TRAIN_COLOR, label="Train", linewidth=1.8, marker="o", markersize=3)
    ax_top5.plot(val.index, val["acc_top5"] * 100, color=VAL_COLOR, label="Val", linewidth=1.8, marker="s", markersize=3)
    if not test.empty and "acc_top5" in test.columns:
        val_acc5 = test["acc_top5"].iloc[0] * 100
        ax_top5.axhline(y=val_acc5, color=TEST_COLOR, linestyle="--", linewidth=1.5, label=f"Test ({val_acc5:.2f}%)")
    _style_ax(ax_top5, "Top-5 Accuracy", "Accuracy (%)")

    if show:
        plt.tight_layout()
        plt.show()
    return ax_top1, ax_top5


def plot_precision_recall_f1(
    metrics: dict[str, pd.DataFrame],
    axes: tuple[plt.Axes, plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Axes, ...]:
    """Plot Precision, Recall, and F1 score curves."""
    show = axes is None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    train, val, test = metrics["train"], metrics["val"], metrics["test"]
    metric_names = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 Score")]

    for ax, (key, title) in zip(axes, metric_names):
        if key in train.columns:
            ax.plot(train.index, train[key], color=TRAIN_COLOR, label="Train", linewidth=1.8, marker="o", markersize=3)
        if key in val.columns:
            ax.plot(val.index, val[key], color=VAL_COLOR, label="Val", linewidth=1.8, marker="s", markersize=3)
        if not test.empty and key in test.columns:
            test_val = test[key].iloc[0]
            ax.axhline(y=test_val, color=TEST_COLOR, linestyle="--", linewidth=1.5, label=f"Test ({test_val:.4f})")
        _style_ax(ax, title, title)

    if show:
        plt.tight_layout()
        plt.show()
    return axes


def plot_train_val_gap(
    metrics: dict[str, pd.DataFrame],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the gap between train and val accuracy to visualize overfitting."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    train, val = metrics["train"], metrics["val"]

    # Align on common epochs
    common_epochs = train.index.intersection(val.index)
    gap = (train.loc[common_epochs, "acc_top1"] - val.loc[common_epochs, "acc_top1"]) * 100

    ax.fill_between(common_epochs, 0, gap, alpha=0.3, color="#E91E63", label="Train - Val gap")
    ax.plot(common_epochs, gap, color="#E91E63", linewidth=1.8, marker="D", markersize=3)
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
    _style_ax(ax, "Overfitting Gap (Top-1 Accuracy)", "Gap (pp)")
    ax.set_ylabel("Train - Val Accuracy (pp)", fontsize=11)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


# ── Dashboard ───────────────────────────────────────────────────────────────

def plot_dashboard(
    csv_path: str | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Generate a comprehensive 3x2 dashboard of training metrics.

    Layout:
        Row 1:  Loss            | Overfitting Gap
        Row 2:  Top-1 Accuracy  | Top-5 Accuracy
        Row 3:  Precision       | Recall & F1

    Parameters
    ----------
    csv_path : str, optional
        Path to the metrics.csv file. Auto-detected if None.
    save_path : str, optional
        Where to save the figure. If None, saves as 'metrics.png' next to the CSV.
    show : bool
        Whether to call plt.show() (useful for interactive use).
    """
    if csv_path is None:
        csv_path = _find_latest_metrics_csv()

    metrics = load_metrics(csv_path)

    fig = plt.figure(figsize=(16, 18), facecolor="white")
    fig.suptitle("Training Dashboard", fontsize=18, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(4, 2, hspace=0.38, wspace=0.28,
                          left=0.07, right=0.97, top=0.93, bottom=0.04)

    # Row 1 ──────────────────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_gap  = fig.add_subplot(gs[0, 1])
    plot_loss(metrics, ax=ax_loss)
    plot_train_val_gap(metrics, ax=ax_gap)

    # Row 2 ──────────────────────────────────────────────────────────────
    ax_top1 = fig.add_subplot(gs[1, 0])
    ax_top5 = fig.add_subplot(gs[1, 1])
    plot_accuracy(metrics, ax_top1=ax_top1, ax_top5=ax_top5)

    # Row 3 ──────────────────────────────────────────────────────────────
    ax_prec = fig.add_subplot(gs[2, 0])
    ax_rec  = fig.add_subplot(gs[2, 1])
    train, val, test = metrics["train"], metrics["val"], metrics["test"]

    for ax, key, title in [(ax_prec, "precision", "Precision"), (ax_rec, "recall", "Recall")]:
        if key in train.columns:
            ax.plot(train.index, train[key], color=TRAIN_COLOR, label="Train", linewidth=1.8, marker="o", markersize=3)
        if key in val.columns:
            ax.plot(val.index, val[key], color=VAL_COLOR, label="Val", linewidth=1.8, marker="s", markersize=3)
        if not test.empty and key in test.columns:
            tv = test[key].iloc[0]
            ax.axhline(y=tv, color=TEST_COLOR, linestyle="--", linewidth=1.5, label=f"Test ({tv:.4f})")
        _style_ax(ax, title, title)

    # Row 4 – F1 (left) + summary table (right) ─────────────────────────
    ax_f1 = fig.add_subplot(gs[3, 0])
    key, title = "f1", "F1 Score"
    if key in train.columns:
        ax_f1.plot(train.index, train[key], color=TRAIN_COLOR, label="Train", linewidth=1.8, marker="o", markersize=3)
    if key in val.columns:
        ax_f1.plot(val.index, val[key], color=VAL_COLOR, label="Val", linewidth=1.8, marker="s", markersize=3)
    if not test.empty and key in test.columns:
        tv = test[key].iloc[0]
        ax_f1.axhline(y=tv, color=TEST_COLOR, linestyle="--", linewidth=1.5, label=f"Test ({tv:.4f})")
    _style_ax(ax_f1, title, title)

    # Summary table
    ax_tbl = fig.add_subplot(gs[3, 1])
    ax_tbl.axis("off")
    _draw_summary_table(ax_tbl, metrics)

    # Save ────────────────────────────────────────────────────────────────
    if save_path is None:
        save_path = os.path.join(os.path.dirname(csv_path), "metrics.png")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Dashboard saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_summary_table(ax: plt.Axes, metrics: dict[str, pd.DataFrame]):
    """Draw a summary table with best/final metrics on the given axes."""
    train, val, test = metrics["train"], metrics["val"], metrics["test"]

    rows = []
    row_labels = []

    # Best validation loss epoch
    if not val.empty and "loss" in val.columns:
        best_epoch = int(val["loss"].idxmin())
        row_labels.append("Best Val Epoch")
        rows.append([str(best_epoch)])

    # Final train metrics
    if not train.empty:
        last = train.iloc[-1]
        row_labels.append("Final Train Loss")
        rows.append([f"{last.get('loss', float('nan')):.4f}"])
        row_labels.append("Final Train Acc@1")
        rows.append([f"{last.get('acc_top1', float('nan')) * 100:.2f}%"])

    # Best val metrics
    if not val.empty:
        best_idx = val["loss"].idxmin()
        best = val.loc[best_idx]
        row_labels.append("Best Val Loss")
        rows.append([f"{best.get('loss', float('nan')):.4f}"])
        row_labels.append("Best Val Acc@1")
        rows.append([f"{best.get('acc_top1', float('nan')) * 100:.2f}%"])
        row_labels.append("Best Val Acc@5")
        rows.append([f"{best.get('acc_top5', float('nan')) * 100:.2f}%"])

    # Test metrics
    if not test.empty:
        t = test.iloc[0]
        row_labels.append("Test Loss")
        rows.append([f"{t.get('loss', float('nan')):.4f}"])
        row_labels.append("Test Acc@1")
        rows.append([f"{t.get('acc_top1', float('nan')) * 100:.2f}%"])
        row_labels.append("Test Acc@5")
        rows.append([f"{t.get('acc_top5', float('nan')) * 100:.2f}%"])
        row_labels.append("Test F1")
        rows.append([f"{t.get('f1', float('nan')):.4f}"])

    if not rows:
        ax.text(0.5, 0.5, "No summary data", ha="center", va="center",
                fontsize=12, color="gray", transform=ax.transAxes)
        return

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=["Value"],
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.8, 1.6)

    # Style header
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif "Test" in row_labels[r - 1]:
            cell.set_facecolor("#E8F5E9")
        elif "Best" in row_labels[r - 1]:
            cell.set_facecolor("#FFF3E0")
        else:
            cell.set_facecolor("#E3F2FD")

    ax.set_title("Summary", fontsize=13, fontweight="bold", pad=10)


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Lightning training metrics")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to metrics.csv (auto-detected if omitted)")
    parser.add_argument("--save", type=str, default=None,
                        help="Output path for the dashboard image")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively")
    args = parser.parse_args()

    plot_dashboard(csv_path=args.csv, save_path=args.save, show=args.show)
