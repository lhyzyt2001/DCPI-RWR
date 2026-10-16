from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "RWR-STRING": "STRING-RWR",
    "Adaptive-supra performance": "Adaptive supra\n(performance)",
    "Adaptive-supra bias-aware": "Adaptive supra\n(bias-aware)",
    "Adaptive-supra evidence-weighted": "Adaptive supra\n(evidence-weighted)",
    "Degree-propensity-weighted bagged PU stack": "Propensity-proxy\nPU stack",
}
COLORS = {
    "RWR-STRING": "#0072B2",
    "Adaptive-supra performance": "#D55E00",
    "Adaptive-supra bias-aware": "#009E73",
    "Adaptive-supra evidence-weighted": "#E69F00",
    "Degree-propensity-weighted bagged PU stack": "#CC79A7",
}
MARKERS = {
    "RWR-STRING": "o",
    "Adaptive-supra performance": "s",
    "Adaptive-supra bias-aware": "^",
    "Adaptive-supra evidence-weighted": "D",
    "Degree-propensity-weighted bagged PU stack": "X",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    adaptive = pd.read_csv(args.results / "01_adaptive_supra" / "adaptive_supra_cross_release_metrics.csv")
    pu = pd.read_csv(args.results / "03_propensity_pu" / "propensity_pu_cross_release_metrics.csv")
    data = pd.concat([adaptive, pu], ignore_index=True, sort=False)
    inference = pd.read_csv(args.results / "05_clustered_inference" / "top100_clustered_inference.csv")
    inference = inference[inference.cluster == "gene"].copy()
    summary = data.groupby("method", as_index=False).agg(
        mean_AUPRC=("AUPRC", "mean"),
        mean_hub=("top100_hub_fraction", "mean"),
        pooled_hits=("top100_hits", "sum"),
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.35, 1.0, 1.12]})
    cancers = ["LUAD", "BRCA", "CRC", "PRAD", "MEL", "OV"]
    methods = list(LABELS)
    for method in methods:
        block = data[data.method == method].set_index("cancer").reindex(cancers)
        axes[0].plot(
            cancers,
            block.AUPRC,
            color=COLORS[method],
            marker=MARKERS[method],
            ms=3.2,
            lw=1.0,
            label=LABELS[method].replace("\n", " "),
        )
    axes[0].set_ylabel("Cross-release AUPRC")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", color="#E6E6E6", lw=0.5)

    for _, row in summary.iterrows():
        method = row.method
        axes[1].scatter(
            row.mean_hub,
            row.mean_AUPRC,
            s=34,
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axes[1].annotate(str(int(row.pooled_hits)), (row.mean_hub, row.mean_AUPRC), xytext=(3, 3), textcoords="offset points", fontsize=6)
    axes[1].set_xlabel("Mean Top-100 hub fraction")
    axes[1].set_ylabel("Mean cross-release AUPRC")
    axes[1].grid(color="#E6E6E6", lw=0.5)
    axes[1].text(0.02, 0.02, "Labels: pooled Top-100 hits", transform=axes[1].transAxes, fontsize=5.8, va="bottom")

    short_ci = {
        "Adaptive-supra performance": "Supra performance",
        "Adaptive-supra bias-aware": "Supra bias-aware",
        "Adaptive-supra evidence-weighted": "Supra evidence-weighted",
        "Degree-propensity-weighted bagged PU stack": "Propensity-proxy PU",
    }
    inference["label"] = inference.method.map(short_ci)
    inference = inference.sort_values("observed_difference")
    y = np.arange(len(inference))
    for idx, row in enumerate(inference.itertuples()):
        axes[2].errorbar(
            row.observed_difference,
            idx,
            xerr=[[row.observed_difference - row.ci_lower], [row.ci_upper - row.observed_difference]],
            fmt=MARKERS[row.method],
            color=COLORS[row.method],
            ecolor="#666666",
            elinewidth=0.8,
            capsize=2,
            ms=4,
        )
    axes[2].axvline(0, color="#333333", ls="--", lw=0.7)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("Top-100 incident-hit difference\nvs STRING-RWR (95% CI)")
    axes[2].grid(axis="x", color="#E6E6E6", lw=0.5)

    for label, ax in zip("abc", axes):
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.34, left=0.08, right=0.99, top=0.94, wspace=0.48)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.savefig(args.output / "Figure_fivepoint_upgrade.pdf", bbox_inches="tight")
    fig.savefig(args.output / "Figure_fivepoint_upgrade.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
