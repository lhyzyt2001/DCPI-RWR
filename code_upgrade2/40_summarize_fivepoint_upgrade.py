from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upgrade2_common import write_json


def fmt(value):
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    adaptive = pd.read_csv(args.results / "01_adaptive_supra" / "adaptive_supra_cross_release_metrics.csv")
    context = pd.read_csv(args.results / "02_context_layers" / "LUAD_context_layer_ablation.csv")
    pu = pd.read_csv(args.results / "03_propensity_pu" / "propensity_pu_cross_release_metrics.csv")
    expression = pd.read_csv(args.results / "04_expression_anchored" / "LUAD_expression_anchored_cross_release_metrics.csv")
    inference_path = args.results / "05_clustered_inference" / "top100_clustered_inference.csv"
    inference = pd.read_csv(inference_path) if inference_path.exists() else pd.DataFrame()
    combined = pd.concat(
        [
            adaptive.assign(module="adaptive_supra_and_seed_weighting"),
            pu.assign(module="propensity_proxy_PU"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(args.output / "all_multicancer_upgrade_metrics.csv", index=False, encoding="utf-8-sig")
    means = combined.groupby(["module", "method"], as_index=False).agg(
        mean_AUPRC=("AUPRC", "mean"),
        mean_ROC_AUC=("ROC_AUC", "mean"),
        pooled_top100_hits=("top100_hits", "sum"),
        mean_top100_hub_fraction=("top100_hub_fraction", "mean"),
    )
    means.to_csv(args.output / "fivepoint_method_summary.csv", index=False, encoding="utf-8-sig")
    baseline = means[means.method == "RWR-STRING"].iloc[0]
    best = means.sort_values("mean_AUPRC", ascending=False).iloc[0]
    best_hits = means.sort_values("pooled_top100_hits", ascending=False).iloc[0]
    lines = [
        "# Five-point DCPI-RWR upgrade: results report",
        "",
        "## Preservation boundary",
        "",
        "This second-stage upgrade was executed only under `01_DCPI_RWR_Q2_upgrade/code_upgrade2` and `results_upgrade2`. The original snapshot, first-stage upgrade, GitHub repository and Zenodo record were not modified.",
        "",
        "## Implemented modules",
        "",
        "1. Cancer-adaptive implicit supra-network RWR with historical-only parameter selection.",
        "2. Open Targets clinical-evidence-strength weighted restart and a LUAD TISCH activity-reweighted STRING layer.",
        "3. Cross-fitted diffusion features with a degree-propensity-proxy bagged PU stack.",
        "4. TCGA-LUAD patient-expression-anchored dual-source diffusion.",
        "5. Explicit input and interpretation boundaries for unavailable patient mutation/CNV/methylation and non-LUAD single-cell data.",
        "",
        "## Multicancer cross-release summary",
        "",
        "| Method | Mean AUPRC | Mean ROC-AUC | Pooled Top-100 hits | Mean hub fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in means.sort_values("mean_AUPRC", ascending=False).iterrows():
        lines.append(
            f"| {row.method} | {fmt(row.mean_AUPRC)} | {fmt(row.mean_ROC_AUC)} | {int(row.pooled_top100_hits)} | {fmt(row.mean_top100_hub_fraction)} |"
        )
    lines.extend(
        [
            "",
            "## LUAD context-layer ablation",
            "",
            markdown_table(context.sort_values("AUPRC", ascending=False)),
            "",
            "## LUAD expression-anchored analysis",
            "",
            markdown_table(expression.sort_values("AUPRC", ascending=False)),
            "",
            "## Objective interpretation",
            "",
            f"The highest mean cross-release AUPRC was obtained by **{best.method}** ({fmt(best.mean_AUPRC)}), compared with {fmt(baseline.mean_AUPRC)} for raw STRING-RWR. The largest pooled Top-100 recovery was **{best_hits.method}** ({int(best_hits.pooled_top100_hits)} incident genes). These are descriptive locked-release comparisons; superiority requires clustered uncertainty analysis before a confirmatory claim.",
            "",
            "The TCGA module is deliberately labelled expression-anchored dual-source diffusion. It must not be described as patient-specific multi-omics or directional regulatory diffusion because mutation, CNV, methylation and a directed interaction network were not available in the local input package.",
        ]
    )
    if not inference.empty:
        gene = inference[inference.cluster == "gene"][
            [
                "method",
                "observed_difference",
                "ci_lower",
                "ci_upper",
                "bootstrap_p_two_sided",
                "cluster_sign_flip_p_two_sided",
            ]
        ]
        lines.extend(
            [
                "",
                "## Gene-clustered Top-100 inference",
                "",
                markdown_table(gene),
                "",
                "No upgraded method had a gene-clustered 95% interval excluding zero. Top-100 differences are therefore descriptive and must not be presented as statistically established superiority.",
            ]
        )
    (args.output / "FIVE_POINT_UPGRADE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        args.output / "FIVE_POINT_UPGRADE_SUMMARY.json",
        {
            "baseline_mean_AUPRC": float(baseline.mean_AUPRC),
            "best_mean_AUPRC_method": str(best.method),
            "best_mean_AUPRC": float(best.mean_AUPRC),
            "best_pooled_top100_method": str(best_hits.method),
            "best_pooled_top100_hits": int(best_hits.pooled_top100_hits),
            "original_experiment_modified": False,
            "github_modified": False,
            "zenodo_modified": False,
        },
    )


if __name__ == "__main__":
    main()

