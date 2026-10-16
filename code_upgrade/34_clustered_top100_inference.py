from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from q2_upgrade_common import (
    SEED,
    degree_bins,
    load_module,
    publication_style,
    read_binary_labels,
    read_mapping,
    transition,
    write_json,
)


def two_sided(values: np.ndarray) -> float:
    return float(min(1.0, 2.0 * min((np.sum(values <= 0) + 1) / (len(values) + 1), (np.sum(values >= 0) + 1) / (len(values) + 1))))


def bootstrap_sum(cluster_values: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    n = len(cluster_values)
    output = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 250):
        stop = min(start + 250, replicates)
        indices = rng.integers(0, n, size=(stop - start, n))
        output[start:stop] = cluster_values[indices].sum(axis=1)
    return output


def sign_flip(cluster_values: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    nonzero = cluster_values[cluster_values != 0]
    if len(nonzero) == 0:
        return np.zeros(replicates, dtype=float)
    output = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        signs = rng.choice((-1.0, 1.0), size=(stop - start, len(nonzero)))
        output[start:stop] = signs @ nonzero
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Gene- and cancer-clustered inference for cross-release Top-100 recovery.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--null-permutations", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 3400)

    scores = pd.read_csv(args.scores)
    methods = list(dict.fromkeys(scores.method.astype(str)))
    reference = "RWR-STRING"
    membership_rows = []
    for (cancer, method), block in scores.groupby(["cancer", "method"], sort=False):
        block = block.sort_values(["score", "node_id"], ascending=[False, True]).copy()
        block["top100"] = 0
        block.iloc[: min(100, len(block)), block.columns.get_loc("top100")] = 1
        membership_rows.append(block[["cancer", "method", "node_id", "Gene", "incident_26.06", "top100"]])
    membership = pd.concat(membership_rows, ignore_index=True)
    membership.to_csv(args.output / "top100_membership_by_cancer_and_method.csv.gz", index=False, compression="gzip")

    summary_rows, replicate_rows = [], []
    reference_block = membership[membership.method == reference][
        ["cancer", "node_id", "Gene", "incident_26.06", "top100"]
    ].rename(columns={"top100": "reference_top100"})
    for method in methods:
        if method == reference:
            continue
        alternative = membership[membership.method == method][["cancer", "node_id", "top100"]].rename(columns={"top100": "method_top100"})
        paired = reference_block.merge(alternative, on=["cancer", "node_id"], validate="one_to_one")
        paired["hit_delta"] = paired["incident_26.06"] * (paired.method_top100 - paired.reference_top100)
        by_gene = paired.groupby(["node_id", "Gene"], as_index=False).hit_delta.sum()
        by_cancer = paired.groupby("cancer", as_index=False).hit_delta.sum()
        gene_boot = bootstrap_sum(by_gene.hit_delta.to_numpy(dtype=float), args.replicates, rng)
        cancer_boot = bootstrap_sum(by_cancer.hit_delta.to_numpy(dtype=float), args.replicates, rng)
        flips = sign_flip(by_gene.hit_delta.to_numpy(dtype=float), args.replicates, rng)
        observed = float(paired.hit_delta.sum())
        sign_p = float((1 + np.sum(np.abs(flips) >= abs(observed))) / (len(flips) + 1))
        summary_rows.extend(
            [
                {
                    "method": method,
                    "reference": reference,
                    "estimand": "pooled six-cancer Top-100 incident-hit difference",
                    "observed_difference": observed,
                    "cluster": "gene",
                    "bootstrap_median": float(np.median(gene_boot)),
                    "ci_lower": float(np.quantile(gene_boot, 0.025)),
                    "ci_upper": float(np.quantile(gene_boot, 0.975)),
                    "bootstrap_p_two_sided": two_sided(gene_boot),
                    "cluster_sign_flip_p_two_sided": sign_p,
                    "clusters": len(by_gene),
                    "replicates": args.replicates,
                },
                {
                    "method": method,
                    "reference": reference,
                    "estimand": "pooled six-cancer Top-100 incident-hit difference",
                    "observed_difference": observed,
                    "cluster": "cancer",
                    "bootstrap_median": float(np.median(cancer_boot)),
                    "ci_lower": float(np.quantile(cancer_boot, 0.025)),
                    "ci_upper": float(np.quantile(cancer_boot, 0.975)),
                    "bootstrap_p_two_sided": two_sided(cancer_boot),
                    "cluster_sign_flip_p_two_sided": np.nan,
                    "clusters": len(by_cancer),
                    "replicates": args.replicates,
                },
            ]
        )
        for index, value in enumerate(gene_boot):
            replicate_rows.append({"method": method, "cluster": "gene", "replicate": index, "hit_difference": value})
        for index, value in enumerate(cancer_boot):
            replicate_rows.append({"method": method, "cluster": "cancer", "replicate": index, "hit_difference": value})
        print(f"{method}: clustered Top-100 inference complete", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "top100_clustered_inference.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(replicate_rows).to_csv(args.output / "top100_clustered_bootstrap_replicates.csv.gz", index=False, compression="gzip")

    null_summary = None
    if args.null_permutations > 0:
        root = args.package_root
        core = load_module(root / "code" / "25_degree_conditioned_pathway_rwr.py", "dcpi_cluster")
        data = torch.load(root / "inputs" / "GNN_Dataset_Raw.pt", map_location="cpu", weights_only=False)
        mapping, symbols = read_mapping(root / "inputs" / "node_mapping.json")
        size = len(symbols)
        transitions, degrees = [], []
        for edge_name, weight_name in (
            ("edge_index_ppi", "edge_weight_ppi"),
            ("edge_index_coexp", "edge_weight_coexp"),
            ("edge_index_pathway", "edge_weight_pathway"),
        ):
            edge = getattr(data, edge_name).cpu().numpy()
            weight = getattr(data, weight_name).cpu().numpy().astype(float)
            transitions.append(transition(edge, weight, size))
            degrees.append(np.bincount(edge[0], minlength=size).astype(float))
        bins = [degree_bins(degree) for degree in degrees]
        configs = json.loads((root / "results" / "21_constrained_dcpi_rwr" / "DCPI_RWR_selected_configs.json").read_text(encoding="utf-8"))
        selected = configs["LUAD"]["bias_aware"]
        cfg = core.Config(float(selected["restart"]), float(selected["blend"]), float(selected["weight_string"]), float(selected["weight_celltype"]), float(selected["weight_pathway"]))
        historical = read_binary_labels(root / "results" / "01_labels" / "LUAD" / "historical_26.03.csv", mapping, size)
        current = read_binary_labels(root / "results" / "01_labels" / "LUAD" / "future_26.06.csv", mapping, size)
        null, null_summary = core.degree_matched_seed_null(historical, current, transitions, bins, bins[0], cfg, args.null_permutations)
        null.to_csv(args.output / "LUAD_degree_matched_seed_null_high_resolution.csv.gz", index=False, compression="gzip")
        write_json(args.output / "LUAD_degree_matched_seed_null_high_resolution_summary.json", null_summary)

    write_json(
        args.output / "clustered_inference_manifest.json",
        {
            "primary_unit": "unique gene; all cancer-specific Top-100 contributions for a sampled gene are retained together",
            "secondary_unit": "cancer; all genes and methods for a sampled cancer are retained together",
            "paired_estimand": "alternative minus RWR-STRING incident hits across six cancer-specific Top-100 lists",
            "tie_break": "descending score, then ascending node identifier",
            "replicates": args.replicates,
            "LUAD_degree_matched_seed_null_replicates": args.null_permutations,
            "random_seed": SEED + 3400,
        },
    )

    publication_style()
    import matplotlib.pyplot as plt

    gene = summary[summary.cluster == "gene"].sort_values("observed_difference")
    fig, ax = plt.subplots(figsize=(6.8, max(2.8, 0.34 * len(gene) + 1.0)))
    y = np.arange(len(gene))
    ax.errorbar(
        gene.observed_difference,
        y,
        xerr=np.vstack([gene.observed_difference - gene.ci_lower, gene.ci_upper - gene.observed_difference]),
        fmt="o",
        color="#0072B2",
        ecolor="#777777",
        capsize=2,
        lw=1,
    )
    ax.axvline(0, color="#333333", lw=0.8, ls="--")
    ax.set_yticks(y, gene.method)
    ax.set_xlabel("Top-100 incident-hit difference vs RWR-STRING (gene-clustered 95% CI)")
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    fig.savefig(args.output / "top100_gene_clustered_inference.pdf", bbox_inches="tight")
    fig.savefig(args.output / "top100_gene_clustered_inference.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
