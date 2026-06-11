"""
analyze_results.py
==================
Produces four individual matplotlib figures from experiment_results.csv.

All shapes in shapes_to_validate/ are ground-truth VALID, so accuracy is
defined as the fraction of queries where the model correctly responds "valid".

Charts
------
1. Parse success rate vs. accuracy per model  (grouped bar)
2. Accuracy by constraint order (0th / 1st / 2nd) grouped by model  (grouped bar)
3. Accuracy heatmap: model x equation  (heatmap)
4. Selected-equation heatmap with count labels  (heatmap)

Output
------
figures/01_parse_success_vs_accuracy_per_model.png / .pdf
figures/03_accuracy_by_order.png / .pdf
figures/04_accuracy_heatmap.png / .pdf
figures/05_accuracy_heatmap_selected_equations.png / .pdf
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

matplotlib.rcParams.update({"font.size": 9})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join("results", "experiment_results.csv")
SHAPES_DIR = "shapes_to_validate"
FIGURES_DIR = "figures"

# ---------------------------------------------------------------------------
# Model display names (shorter labels for plots)
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "mathstral:7b":    "Mathstral\n7B",
    "ministral-3:8b":  "Ministral-3\n8B",
    "gemma4:e2b":      "Gemma4\ne2b",
    "glm4:9b":         "GLM4\n9B",
}

# Consistent colour per model
MODEL_COLOURS = {
    "mathstral:7b":    "#2196F3",   # blue  (math-specialised)
    "ministral-3:8b":  "#90CAF9",   # light blue
    "gemma4:e2b":      "#4CAF50",   # green (latest-gen general)
    "glm4:9b":         "#9C27B0",   # purple
}

# ---------------------------------------------------------------------------
# Constraint-order inference
# ---------------------------------------------------------------------------
_SHAPE_ORDER_CACHE = {}  # type: dict


def _infer_constraint_order(equation_name: str, shape_id: int) -> int:
    """Return 0, 1, or 2 based on the derivative order in the shape file."""
    key = (equation_name, shape_id)
    if key in _SHAPE_ORDER_CACHE:
        return _SHAPE_ORDER_CACHE[key]

    path = os.path.join(SHAPES_DIR, equation_name, f"shape_{shape_id}.md")
    order = -1
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        # Extract the shape property line to avoid matching the example
        prop_match = re.search(
            r"\*\*Shape Property:\*\*\s*\$(.*?)\$", content, re.DOTALL
        )
        prop = prop_match.group(1) if prop_match else ""
        if r"\partial^2" in prop:
            order = 2
        elif r"\partial" in prop:
            order = 1
        else:
            order = 0
    _SHAPE_ORDER_CACHE[key] = order
    return order


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(CSV_PATH)

    # Ground truth: all shapes are valid
    df["is_correct"] = df["result"] == "valid"
    df["is_valid_call"] = df["result"] == "valid"
    # Constraint order
    df["constraint_order"] = df.apply(
        lambda r: _infer_constraint_order(r["equation_name"], int(r["shape_id"])),
        axis=1,
    )
    df["order_label"] = df["constraint_order"].map(
        {0: "0th order\n(non-negativity)",
         1: "1st order\n(monotonicity)",
         2: "2nd order\n(curvature)"}
    )

    df = df[df['model'].isin(["mathstral:7b","ministral-3:8b","gemma4:e2b","glm4:9b"])]


    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _model_order(df):
    """Models sorted by descending accuracy."""
    acc = df.groupby("model")["is_correct"].mean()
    return acc.sort_values(ascending=False).index.tolist()


def _save_figure(fig, name):
    """Save *fig* as both PNG and PDF in FIGURES_DIR."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def plot_parse_success_vs_accuracy_per_model(df):
    fig, ax = plt.subplots(figsize=(7, 3))
    models = _model_order(df)
    labels = [MODEL_LABELS.get(m, m) for m in models]
    parse_rates = []
    accs = []
    for m in models:
        sub = df[df["model"] == m]
        success_norm = sub["success"].astype(str).str.strip().str.lower()
        parse_rates.append(success_norm.isin({"true", "1", "1.0", "yes"}).mean() * 100)
        accs.append(sub["is_correct"].mean() * 100)

    x = np.arange(len(models))
    width = 0.36
    bars_parse = ax.bar(
        x - width / 2,
        parse_rates,
        width=width,
        label="Parse success",
        color="#42A5F5",
        edgecolor="white",
    )
    bars_acc = ax.bar(
        x + width / 2,
        accs,
        width=width,
        label="Accuracy",
        color="#66BB6A",
        edgecolor="white",
    )

    for bar, rate in zip(bars_parse, parse_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar, acc in zip(bars_acc, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_figure(fig, "01_parse_success_vs_accuracy_per_model")



def plot_accuracy_by_order(df):
    fig, ax = plt.subplots(figsize=(7, 3))
    models = _model_order(df)
    orders = [0, 1, 2]
    order_names = ["0th order", "1st order", "2nd order"]
    order_colours = ["#1976D2", "#43A047", "#E53935"]

    x = np.arange(len(models))
    n_orders = len(orders)
    width = 0.22
    offsets = np.linspace(-(n_orders - 1) / 2, (n_orders - 1) / 2, n_orders) * width

    for offset, order, name, colour in zip(offsets, orders, order_names, order_colours):
        accs = []
        for m in models:
            sub = df[(df["model"] == m) & (df["constraint_order"] == order)]
            accs.append(sub["is_correct"].mean() * 100 if len(sub) > 0 else 0.0)
        bars = ax.bar(
            x + offset,
            accs,
            width=width,
            label=name,
            color=colour,
            edgecolor="white",
            alpha=0.9,
        )
        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_figure(fig, "03_accuracy_by_order")


def plot_heatmap_model_equation(df):
    models = _model_order(df)
    equations = sorted(df["equation_name"].unique())
    matrix = np.full((len(models), len(equations)), np.nan)

    for i, m in enumerate(models):
        for j, eq in enumerate(equations):
            sub = df[(df["model"] == m) & (df["equation_name"] == eq)]
            if len(sub) > 0:
                matrix[i, j] = sub["is_correct"].mean()

    fig_width = max(10, len(equations) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 3))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in models],
                       fontsize=7)
    ax.set_xticks(range(len(equations)))
    ax.set_xticklabels(equations, rotation=90, fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Accuracy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_figure(fig, "04_accuracy_heatmap")


def plot_heatmap_selected_equations(df):
    models = _model_order(df)
    equations = [
        "FeynmanICh9Eq18",
        "FeynmanICh30Eq3",
        "FeynmanICh41Eq16",
        "FeynmanIIICh4Eq33",
    ]
    matrix = np.full((len(models), len(equations)), np.nan)
    counts = [[(0, 0, 0) for _ in equations] for _ in models]

    for i, m in enumerate(models):
        for j, eq in enumerate(equations):
            sub = df[(df["model"] == m) & (df["equation_name"] == eq)]
            if len(sub) > 0:
                matrix[i, j] = sub["is_correct"].mean()
            correct = int(sub["is_correct"].sum())
            success_norm = sub["success"].astype(str).str.strip().str.lower()
            parse_success = int(success_norm.isin({"true", "1", "1.0", "yes"}).sum())
            total = int(len(sub))
            counts[i][j] = (correct, parse_success, total)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in models],
                       fontsize=7)
    ax.set_xticks(range(len(equations)))
    ax.set_xticklabels(equations, rotation=30, ha="right", fontsize=7)

    # Show absolute counts as correct / parse-success / total inside each cell.
    norm = mcolors.Normalize(vmin=0, vmax=1)
    for i in range(len(models)):
        for j in range(len(equations)):
            correct, parse_success, total = counts[i][j]
            if total > 0 and not np.isnan(matrix[i, j]):
                pct = matrix[i, j] * 100
                rgba = cmap(norm(matrix[i, j]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_colour = "black" if luminance > 0.6 else "white"
                label = f"{correct} / {parse_success} / {total}\n{pct:.0f}%"
            else:
                text_colour = "black"
                label = "0 / 0 / 0\nN/A"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=text_colour,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04, label="Accuracy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_figure(fig, "05_accuracy_heatmap_selected_equations")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading data from {CSV_PATH} ...")
    df = load_data()

    print(f"  Rows: {len(df)}  |  Models: {df['model'].nunique()}  "
          f"|  Equations: {df['equation_name'].nunique()}")
    print(f"  Constraint orders found: {sorted(df['constraint_order'].unique())}")
    print(f"\nSaving figures to {FIGURES_DIR}/ ...")

    plot_parse_success_vs_accuracy_per_model(df)
    # plot_valid_invalid_stacked(df)
    plot_accuracy_by_order(df)
    plot_heatmap_model_equation(df)
    plot_heatmap_selected_equations(df)
    # plot_model_agreement(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
