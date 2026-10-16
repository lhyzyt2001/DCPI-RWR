from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from q2_upgrade_common import (
    CANCERS,
    SEED,
    degree_bins,
    load_module,
    publication_style,
    read_binary_labels,
    read_mapping,
    rwr,
    safe_metrics,
    topk_rows,
    transition,
    write_json,
)


def dcpi_score(core, transitions, bins, seeds, size: int, config: dict) -> np.ndarray:
    cfg = core.Config(
        float(config["restart"]),
        float(config["blend"]),
        float(config["weight_string"]),
        float(config["weight_celltype"]),
        float(config["weight_pathway"]),
    )
    return core.score_config(core.components(transitions, bins, seeds, size, cfg.restart), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-release label-boundary and network-input sensitivity.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancers", nargs="+", choices=CANCERS, default=CANCERS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    root = args.package_root
    core = load_module(root / "code" / "25_degree_conditioned_pathway_rwr.py", "dcpi_boundary")
    data = torch.load(root / "inputs" / "GNN_Dataset_Raw.pt", map_location="cpu", weights_only=False)
    mapping, symbols = read_mapping(root / "inputs" / "node_mapping.json")
    size = len(symbols)
    selected = json.loads(
        (root / "results" / "21_constrained_dcpi_rwr" / "DCPI_RWR_selected_configs.json").read_text(encoding="utf-8")
    )

    string_edge = data.edge_index_ppi.cpu().numpy()
    string_weight = data.edge_weight_ppi.cpu().numpy().astype(float)
    cell_edge = data.edge_index_coexp.cpu().numpy()
    cell_weight = data.edge_weight_coexp.cpu().numpy().astype(float)
    path_edge = data.edge_index_pathway.cpu().numpy()
    path_weight = data.edge_weight_pathway.cpu().numpy().astype(float)
    cell_transition = transition(cell_edge, cell_weight, size)
    path_transition = transition(path_edge, path_weight, size)
    cell_degree = np.bincount(cell_edge[0], minlength=size).astype(float)
    path_degree = np.bincount(path_edge[0], minlength=size).astype(float)

    rng = np.random.default_rng(SEED + 3200)
    scenario_specs: list[tuple[str, np.ndarray]] = [
        ("STRING original packaged graph", np.ones(string_weight.size, dtype=bool))
    ]
    for threshold in (0.80, 0.90):
        scenario_specs.append((f"STRING score >= {threshold:.2f}", string_weight >= threshold))
    all_indices = np.arange(string_weight.size)
    for removal in (0.10, 0.20):
        keep = np.ones(string_weight.size, dtype=bool)
        removed = rng.choice(all_indices, size=int(round(removal * string_weight.size)), replace=False)
        keep[removed] = False
        scenario_specs.append((f"STRING deterministic edge dropout {int(removal * 100)}%", keep))

    churn_rows, metric_rows, topk, rank_rows = [], [], [], []
    scenario_cache = {}
    for scenario, keep in scenario_specs:
        edge = string_edge[:, keep]
        weight = string_weight[keep]
        tr = transition(edge, weight, size)
        deg = np.bincount(edge[0], minlength=size).astype(float)
        scenario_cache[scenario] = (tr, deg, [degree_bins(deg), degree_bins(cell_degree), degree_bins(path_degree)])

    base_transition, base_degree, base_bins = scenario_cache["STRING original packaged graph"]
    for cancer in args.cancers:
        historical = read_binary_labels(root / "results" / "01_labels" / cancer / "historical_26.03.csv", mapping, size)
        current = read_binary_labels(root / "results" / "01_labels" / cancer / "future_26.06.csv", mapping, size)
        hist_nodes = np.flatnonzero(historical == 1)
        persistent_nodes = np.flatnonzero((historical == 1) & (current == 1))
        incident = np.flatnonzero((historical == 0) & (current == 1))
        removed = np.flatnonzero((historical == 1) & (current == 0))
        candidate = np.flatnonzero(historical == 0)
        labels = ((historical[candidate] == 0) & (current[candidate] == 1)).astype(np.int8)
        churn_rows.append(
            {
                "cancer": cancer,
                "historical_positive": len(hist_nodes),
                "persistent_positive": len(persistent_nodes),
                "removed_by_26.06": len(removed),
                "incident_by_26.06": len(incident),
                "historical_retention_fraction": len(persistent_nodes) / max(len(hist_nodes), 1),
            }
        )
        cfg = selected[cancer]["performance"]
        original_scores = {}
        for seed_label, seeds in (("historical seeds", hist_nodes), ("persistent seeds (diagnostic)", persistent_nodes)):
            raw = rwr(base_transition, seeds, size, 0.30)
            dcpi = dcpi_score(core, [base_transition, cell_transition, path_transition], base_bins, seeds, size, cfg)
            for method, full_score in (("RWR-STRING", raw), ("DCPI-RWR performance", dcpi)):
                score = full_score[candidate]
                roc, ap = safe_metrics(labels, score)
                metric_rows.append(
                    {
                        "cancer": cancer,
                        "analysis": "seed-boundary diagnostic",
                        "scenario": seed_label,
                        "method": method,
                        "ROC_AUC": roc,
                        "AUPRC": ap,
                        "score_degree_spearman": float(spearmanr(score, base_degree[candidate]).statistic),
                    }
                )
                topk.extend(topk_rows(cancer, f"{method} | {seed_label}", candidate, labels, score, base_degree))
                if seed_label == "historical seeds":
                    original_scores[method] = full_score

        for scenario, _ in scenario_specs:
            tr, deg, bins = scenario_cache[scenario]
            raw = rwr(tr, hist_nodes, size, 0.30)
            dcpi = dcpi_score(core, [tr, cell_transition, path_transition], bins, hist_nodes, size, cfg)
            for method, full_score in (("RWR-STRING", raw), ("DCPI-RWR performance", dcpi)):
                score = full_score[candidate]
                roc, ap = safe_metrics(labels, score)
                metric_rows.append(
                    {
                        "cancer": cancer,
                        "analysis": "network-input sensitivity",
                        "scenario": scenario,
                        "method": method,
                        "ROC_AUC": roc,
                        "AUPRC": ap,
                        "score_degree_spearman": float(spearmanr(score, deg[candidate]).statistic),
                    }
                )
                topk.extend(topk_rows(cancer, f"{method} | {scenario}", candidate, labels, score, deg))
                rank_rows.append(
                    {
                        "cancer": cancer,
                        "scenario": scenario,
                        "method": method,
                        "spearman_vs_original": float(spearmanr(score, original_scores[method][candidate]).statistic),
                        "retained_directed_edges": int(tr.nnz),
                    }
                )
        print(f"{cancer}: boundary and network sensitivity complete", flush=True)

    churn = pd.DataFrame(churn_rows)
    metrics = pd.DataFrame(metric_rows)
    topk_frame = pd.DataFrame(topk)
    ranks = pd.DataFrame(rank_rows)
    churn.to_csv(args.output / "cross_release_label_churn.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output / "cross_release_boundary_metrics.csv", index=False, encoding="utf-8-sig")
    topk_frame.to_csv(args.output / "cross_release_boundary_topk.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(args.output / "network_input_rank_stability.csv", index=False, encoding="utf-8-sig")
    write_json(
        args.output / "cross_release_boundary_manifest.json",
        {
            "deployment_evaluation": "26.03 labels define seeds and configuration; incident 26.06 labels among historical-unlabelled genes define outcomes.",
            "persistent_seed_warning": "Uses knowledge of 26.06 label persistence and is therefore diagnostic only, not an unbiased deployment evaluation.",
            "network_snapshot_warning": "Historical STRING snapshots were unavailable in the package. Threshold and deterministic-dropout analyses test network-input robustness but do not establish fully prospective network freezing.",
            "random_seed": SEED + 3200,
        },
    )

    publication_style()
    import matplotlib.pyplot as plt

    network = metrics[metrics.analysis == "network-input sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.36})
    colors = {"RWR-STRING": "#0072B2", "DCPI-RWR performance": "#D55E00"}
    scenarios = list(dict.fromkeys(network.scenario))
    for method, color in colors.items():
        block = network[network.method == method].groupby("scenario", sort=False).AUPRC.mean().reindex(scenarios)
        axes[0].plot(range(len(scenarios)), block, marker="o", lw=1.3, ms=3.5, color=color, label=method)
        rb = ranks[ranks.method == method].groupby("scenario", sort=False).spearman_vs_original.mean().reindex(scenarios)
        axes[1].plot(range(len(scenarios)), rb, marker="o", lw=1.3, ms=3.5, color=color, label=method)
    labels_short = [s.replace("STRING ", "").replace("deterministic edge ", "") for s in scenarios]
    for ax in axes:
        ax.set_xticks(range(len(scenarios)), labels_short, rotation=35, ha="right")
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    axes[0].set_ylabel("Mean cross-release AUPRC")
    axes[1].set_ylabel("Mean rank correlation vs original")
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(frameon=False)
    fig.savefig(args.output / "cross_release_boundary_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(args.output / "cross_release_boundary_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
