from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from upgrade2_common import (
    evidence_weights,
    metrics,
    percentile,
    read_labels,
    read_mapping,
    topk_summary,
    transition,
    weighted_rwr,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-aberrant-per-tumor", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.package_root
    mapping, symbols = read_mapping(root / "inputs" / "node_mapping.json")
    size = len(symbols)
    data = torch.load(root / "inputs" / "GNN_Dataset_Raw.pt", map_location="cpu", weights_only=False)
    edge = data.edge_index_ppi.cpu().numpy()
    edge_weight = data.edge_weight_ppi.cpu().numpy().astype(float)
    matrix = transition(edge, edge_weight, size)
    degree = np.bincount(edge[0], minlength=size).astype(float)
    hub_cutoff = float(np.quantile(degree, 0.95))

    expression = pd.read_csv(args.expression, sep="\t", index_col=0)
    expression.index = expression.index.astype(str).str.split("|").str[0]
    expression = expression[~expression.index.duplicated(keep="first")]
    tumor_columns = [c for c in expression.columns if str(c).split("-")[-1][:2] in {"01", "02"}]
    normal_columns = [c for c in expression.columns if str(c).split("-")[-1][:2] == "11"]
    if len(tumor_columns) < 20 or len(normal_columns) < 10:
        raise RuntimeError("TCGA expression matrix does not contain adequate tumour and adjacent-normal samples")
    common = [gene for gene in expression.index if gene in mapping]
    values_t = expression.loc[common, tumor_columns].to_numpy(float)
    values_n = expression.loc[common, normal_columns].to_numpy(float)
    normal_median = np.nanmedian(values_n, axis=1)
    normal_mad = np.nanmedian(np.abs(values_n - normal_median[:, None]), axis=1)
    fallback = np.nanmedian(normal_mad[normal_mad > 0])
    normal_mad[normal_mad <= 0] = fallback if np.isfinite(fallback) and fallback > 0 else 1.0
    z = (values_t - normal_median[:, None]) / (1.4826 * normal_mad[:, None])
    frequency = np.zeros(len(common), dtype=float)
    for column in range(z.shape[1]):
        order = np.argsort(-np.abs(z[:, column]), kind="mergesort")[: args.top_aberrant_per_tumor]
        frequency[order] += 1.0
    frequency /= z.shape[1]
    cohort_effect = np.abs(np.nanmedian(values_t, axis=1) - normal_median)
    downstream_weight = frequency * cohort_effect
    downstream_nodes_local = np.flatnonzero(downstream_weight > 0)
    downstream_nodes = np.asarray([mapping[common[i]] for i in downstream_nodes_local], dtype=int)
    downstream_weights = downstream_weight[downstream_nodes_local]

    historical = read_labels(root / "results" / "01_labels" / "LUAD" / "historical_26.03.csv", mapping, size)
    future = read_labels(root / "results" / "01_labels" / "LUAD" / "future_26.06.csv", mapping, size)
    incident = ((future == 1) & (historical == 0)).astype(np.int8)
    seeds = np.flatnonzero(historical == 1)
    evidence = evidence_weights(root / "results" / "01_labels" / "LUAD" / "26.03_disease_specific_evidence.csv.gz", mapping, size)
    upstream = weighted_rwr(matrix, seeds, size, 0.30, evidence[seeds])
    downstream = weighted_rwr(matrix.T.tocsc(), downstream_nodes, size, 0.30, downstream_weights)
    consensus = np.sqrt(percentile(upstream) * percentile(downstream))
    candidates = np.flatnonzero(historical == 0)
    y = incident[candidates]
    rows = []
    for method, full_score in {
        "Evidence-weighted clinical RWR": upstream,
        "TCGA expression-aberration diffusion": downstream,
        "Expression-anchored dual-source consensus": consensus,
    }.items():
        roc, auprc = metrics(y, full_score[candidates])
        rows.append(
            {
                "method": method,
                "ROC_AUC": roc,
                "AUPRC": auprc,
                **topk_summary(y, full_score[candidates], degree[candidates], hub_cutoff=hub_cutoff),
            }
        )
    pd.DataFrame(rows).to_csv(args.output / "LUAD_expression_anchored_cross_release_metrics.csv", index=False, encoding="utf-8-sig")
    order = candidates[np.argsort(-consensus[candidates], kind="mergesort")[:500]]
    pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "Gene": [symbols[i] for i in order],
            "node_id": order,
            "consensus_score": consensus[order],
            "clinical_RWR": upstream[order],
            "expression_diffusion": downstream[order],
            "incident_26.06": incident[order],
            "STRING_degree": degree[order],
        }
    ).to_csv(args.output / "LUAD_expression_anchored_top500.csv", index=False, encoding="utf-8-sig")
    per_gene = pd.DataFrame(
        {
            "Gene": common,
            "tumor_aberration_frequency": frequency,
            "absolute_tumor_normal_median_difference": cohort_effect,
            "downstream_restart_weight": downstream_weight,
        }
    )
    per_gene.to_csv(args.output / "LUAD_TCGA_expression_restart_weights.csv.gz", index=False, compression="gzip")
    write_json(
        args.output / "expression_anchored_manifest.json",
        {
            "tumor_samples": len(tumor_columns),
            "adjacent_normal_samples": len(normal_columns),
            "mapped_genes": len(common),
            "top_aberrant_genes_per_tumor": args.top_aberrant_per_tumor,
            "boundary": "Patient expression profiles determine a cohort restart distribution. Mutation, CNV and methylation were unavailable, and packaged STRING is effectively undirected; this is an exploratory dual-source consensus, not a full directional NetICS-style multi-omics analysis.",
        },
    )


if __name__ == "__main__":
    main()

