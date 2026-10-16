from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the independent DCPI-RWR Q2 upgrade analyses.")
    parser.add_argument("--upgrade-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.upgrade_root
    results = root / "results_upgrade"

    metrics = pd.read_csv(results / "01_degree_aware_multiplex" / "degree_aware_cross_release_metrics.csv")
    topk = pd.read_csv(results / "01_degree_aware_multiplex" / "degree_aware_topk_and_hubs.csv")
    clustered = pd.read_csv(results / "04_clustered_inference" / "top100_clustered_inference.csv")
    churn = pd.read_csv(results / "02_cross_release_boundary" / "cross_release_label_churn.csv")
    boundary = pd.read_csv(results / "02_cross_release_boundary" / "cross_release_boundary_metrics.csv")
    rank_stability = pd.read_csv(results / "02_cross_release_boundary" / "network_input_rank_stability.csv")
    pu = pd.read_csv(results / "03_pu_sensitivity" / "pu_ascertainment_delta_vs_rwr.csv")
    null_summary = json.loads(
        (results / "04_clustered_inference" / "LUAD_degree_matched_seed_null_high_resolution_summary.json").read_text(encoding="utf-8")
    )

    mean_metrics = metrics.groupby("method", as_index=False).agg(mean_AUPRC=("AUPRC", "mean"), mean_ROC_AUC=("ROC_AUC", "mean"), mean_abs_degree_rho=("score_degree_spearman", lambda x: float(np.mean(np.abs(x)))))
    hub = topk[topk.K == 100].groupby("method", as_index=False).agg(mean_top100_hits=("incident_hits", "mean"), pooled_top100_hits=("incident_hits", "sum"), mean_hub_fraction=("hub_fraction", "mean"))
    aggregate = mean_metrics.merge(hub, on="method").sort_values("mean_AUPRC", ascending=False)
    aggregate.to_csv(results / "Q2_UPGRADE_AGGREGATE_SUMMARY.csv", index=False, encoding="utf-8-sig")

    winners = metrics.loc[metrics.groupby("cancer").AUPRC.idxmax(), ["cancer", "method", "AUPRC"]].sort_values("cancer")
    winner_lines = "\n".join(f"- {row.cancer}: {row.method}, AUPRC {row.AUPRC:.4f}." for row in winners.itertuples())
    aggregate_lines = "\n".join(
        f"| {row.method} | {row.mean_AUPRC:.4f} | {row.mean_ROC_AUC:.4f} | {row.mean_abs_degree_rho:.3f} | {row.pooled_top100_hits:.0f} | {row.mean_hub_fraction:.3f} |"
        for row in aggregate.itertuples()
    )
    cluster_gene = clustered[clustered.cluster == "gene"]
    cluster_lines = "\n".join(
        f"| {row.method} | {row.observed_difference:.0f} | {row.ci_lower:.0f} to {row.ci_upper:.0f} | {row.bootstrap_p_two_sided:.3f} | {row.cluster_sign_flip_p_two_sided:.3f} |"
        for row in cluster_gene.itertuples()
    )
    incomplete = pu[pu.retained_positive_fraction < 1.0]
    pu_summary = (
        incomplete.groupby("method", as_index=False)
        .agg(
            scenarios=("median_delta_AUPRC", "size"),
            positive_median_scenarios=("median_delta_AUPRC", lambda x: int(np.sum(np.asarray(x) > 0))),
            median_delta_across_scenarios=("median_delta_AUPRC", "median"),
        )
        .sort_values("median_delta_across_scenarios", ascending=False)
    )
    pu_lines = "\n".join(
        f"| {row.method} | {row.positive_median_scenarios}/{row.scenarios} | {row.median_delta_across_scenarios:.4f} |"
        for row in pu_summary.itertuples()
    )
    base_network = boundary[(boundary.analysis == "network-input sensitivity") & (boundary.scenario == "STRING original packaged graph")]
    perturbed = boundary[(boundary.analysis == "network-input sensitivity") & (boundary.scenario != "STRING original packaged graph")]
    network_summary = perturbed.groupby(["scenario", "method"], as_index=False).AUPRC.mean()
    network_lines = "\n".join(
        f"| {row.scenario} | {row.method} | {row.AUPRC:.4f} |"
        for row in network_summary.itertuples()
    )
    stability_min = rank_stability.groupby("method").spearman_vs_original.min().to_dict()

    report = f"""# DCPI-RWR independent upgrade: results report

## Scope and preservation boundary

This directory is an independent copy created on 2026-10-13. The original experiment package, manuscript, GitHub repository and Zenodo record were not edited. All new code is under `code_upgrade/`; all new outputs are under `results_upgrade/`.

The upgrade addresses four prespecified priorities: (1) degree-aware and multiplex diffusion comparators; (2) cross-release and network-input boundary sensitivity; (3) positive-unlabelled label-ascertainment sensitivity; and (4) gene- and cancer-clustered Top-100 inference with a higher-resolution LUAD degree-matched seed null.

## 1. Degree-aware and multiplex comparators

The DADA-CD comparator is a documented candidate-degree reference-model reimplementation inspired by Erten et al. (PLoS ONE 2011; doi:10.1371/journal.pone.0021171), not a claim of bit-for-bit reproduction of the historical MATLAB implementation. Multiplex RWR is defined as RWR on a convex mixture of column-normalized layer transition matrices. Hyperparameters were selected using Open Targets 26.03 labels only and evaluated on genes newly labelled by 26.06.

| Method | Mean AUPRC | Mean ROC-AUC | Mean absolute Spearman rho | Pooled Top-100 incident hits | Mean Top-100 hub fraction |
|---|---:|---:|---:|---:|---:|
{aggregate_lines}

Cancer-specific AUPRC winners were:

{winner_lines}

Interpretation: no single upgraded method dominates all cancers. Raw STRING-RWR remains the strongest mean-AUPRC method and wins BRCA, CRC, MEL and PRAD. Multiplex STRING+pathway wins OV, whereas three-view multiplex RWR wins LUAD and has much lower score-degree correlation than raw RWR. This is useful evidence for cancer-specific multiview value, but it does not justify a universal performance-superiority claim.

## 2. Cross-release and network-input boundary

Across cancers, historical-label retention from 26.03 to 26.06 ranged from {churn.historical_retention_fraction.min():.3f} to {churn.historical_retention_fraction.max():.3f}. Removed historical positives were absent in LUAD and ranged up to {int(churn['removed_by_26.06'].max())} in the other cancers. Replacing historical seeds with only labels that persist in 26.06 changed results little in most cancers, but this is explicitly diagnostic because it uses future information.

| Network perturbation | Method | Mean AUPRC across cancers |
|---|---|---:|
{network_lines}

The minimum cancer-specific rank correlation with the original packaged graph was {min(stability_min.values()):.3f}. These analyses support robustness to modest deterministic edge dropout, while stricter STRING thresholds can alter performance. Because a historical STRING snapshot was unavailable, this does not establish fully prospective network freezing.

## 3. Positive-unlabelled label-ascertainment stress test

Predictions were held fixed while 25%, 50% or 75% of incident positives were retained either uniformly (SCAR) or with degree-biased ascertainment. This is a stress test of incomplete outcome ascertainment; it is not a latent-positive estimator.

| Method vs RWR-STRING | Scenarios with positive median delta | Median delta across incomplete-label scenarios |
|---|---:|---:|
{pu_lines}

The method ordering is therefore not uniformly robust to incompletely observed positives. LUAD tends to favour the multiview methods, whereas several other cancers continue to favour raw RWR, especially under degree-biased ascertainment.

## 4. Clustered Top-100 inference

The estimand is the pooled difference in newly labelled genes recovered by six cancer-specific Top-100 lists. The primary bootstrap samples unique genes and keeps all cancer-specific contributions for a sampled gene together; the secondary bootstrap samples cancers.

| Method vs RWR-STRING | Observed hit difference | Gene-clustered 95% CI | Bootstrap P | Cluster sign-flip P |
|---|---:|---:|---:|---:|
{cluster_lines}

DCPI-RWR recovered 10 more pooled Top-100 incident hits than raw RWR, but the gene-clustered interval included zero. Thus the gain is descriptive rather than statistically secure. DADA-CD and the optimistic merge reduced centrality dependence but lost incident-hit recovery.

The 2,000-replicate LUAD degree-matched seed null gave observed AUPRC {null_summary['observed_AUPRC']:.4f}, null mean {null_summary['null_mean_AUPRC']:.4f}, 95% null interval {null_summary['null_ci_lower']:.4f}–{null_summary['null_ci_upper']:.4f}, and empirical upper-tail P={null_summary['empirical_p_greater_equal']:.3f}. This is suggestive but does not pass a conventional 0.05 threshold.

## Overall judgment

The upgrade improves methodological depth and reviewer-facing transparency. Its strongest defensible contribution is not universal superiority over RWR, but a controlled analysis of when multiview diffusion can trade predictive discrimination against network-centrality dependence under cross-release evaluation. The new evidence is internally coherent: some cancers benefit from pathway or three-view integration, while raw RWR remains difficult to beat and the uncertainty of Top-100 gains is now quantified correctly.

For a revised manuscript, retain the negative and heterogeneous results. Do not claim that DCPI-RWR is globally superior, degree independent, prospectively validated, or a causal/therapeutic target predictor. A stronger framing is a degree-aware multiview diffusion benchmarking framework with cancer-specific utility and explicit ascertainment/centrality limits.
"""
    (root / "UPGRADE_RESULTS_REPORT.md").write_text(report, encoding="utf-8")

    run_manifest = {
        "created_date": "2026-10-13",
        "analysis_root": str(root),
        "preserved_snapshot": "not distributed; retained outside the public release",
        "original_GitHub_modified": False,
        "original_Zenodo_modified": False,
        "endpoint": "Open Targets 26.03 seeds/configuration; incident 26.06 evidence among 26.03-unlabelled genes",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib")
        },
        "replicates": {
            "paired_AUPRC_bootstrap": 2000,
            "PU_incomplete_fraction": 1000,
            "Top100_clustered_bootstrap": 10000,
            "LUAD_degree_matched_seed_null": 2000,
        },
    }
    (root / "UPGRADE_RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "UPGRADE_FILE_MANIFEST.sha256.tsv")
    manifest = pd.DataFrame(
        {
            "relative_path": [path.relative_to(root).as_posix() for path in files],
            "bytes": [path.stat().st_size for path in files],
            "sha256": [sha256(path) for path in files],
        }
    )
    manifest.to_csv(root / "UPGRADE_FILE_MANIFEST.sha256.tsv", sep="\t", index=False, encoding="utf-8")
    print(f"Wrote report and manifest for {len(files)} files.")


if __name__ == "__main__":
    main()
