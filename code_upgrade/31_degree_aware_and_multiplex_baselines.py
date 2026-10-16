from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from q2_upgrade_common import (
    CANCERS,
    SEED,
    average_transition,
    conditioned_percentile,
    degree_bins,
    holm_adjust,
    load_module,
    optimistic_uniform_score,
    percentile,
    publication_style,
    read_binary_labels,
    read_mapping,
    rwr,
    safe_metrics,
    topk_rows,
    transition,
    write_json,
)


RESTARTS = (0.15, 0.30, 0.50)
GAMMAS = (0.25, 0.50, 0.75, 1.00)


def dada_candidate_degree_score(raw: np.ndarray, weighted_degree: np.ndarray, window: int = 200) -> np.ndarray:
    """Candidate-degree reference-model reimplementation inspired by DADA-CD.

    Each candidate is compared with a centered window of genes having the closest
    weighted STRING degrees. The empirical upper-tail probability is converted to
    -log10(p). This is documented as a DADA-CD reference-model reimplementation,
    not as the original MATLAB package.
    """

    n = len(raw)
    order = np.argsort(weighted_degree, kind="mergesort")
    inverse = np.empty(n, dtype=int)
    inverse[order] = np.arange(n)
    half = window // 2
    output = np.zeros(n, dtype=float)
    for node in range(n):
        pos = inverse[node]
        left = max(0, min(pos - half, n - window))
        right = min(n, left + window)
        pool = order[left:right]
        count = int(np.count_nonzero(raw[pool] >= raw[node]))
        p_value = (count + 1.0) / (len(pool) + 1.0)
        output[node] = -np.log10(p_value)
    return output


def build_method_scores(
    transitions: list,
    string_degree: np.ndarray,
    weighted_degree: np.ndarray,
    seeds: np.ndarray,
    restart: float,
    cache: dict,
) -> dict[str, np.ndarray]:
    key = (tuple(seeds.tolist()), restart)
    if key not in cache:
        raw_string = rwr(transitions[0], seeds, len(string_degree), restart)
        mix_sp = rwr(average_transition([transitions[0], transitions[2]], [0.5, 0.5]), seeds, len(string_degree), restart)
        mix_all = rwr(average_transition(transitions, [1 / 3, 1 / 3, 1 / 3]), seeds, len(string_degree), restart)
        adjusted = dada_candidate_degree_score(raw_string, weighted_degree)
        cache[key] = {
            "raw": raw_string,
            "dada_cd": adjusted,
            "dada_optimistic": optimistic_uniform_score(raw_string, adjusted),
            "multiplex_sp": mix_sp,
            "multiplex_all": mix_all,
        }
    return cache[key]


def candidate_configs() -> list[dict]:
    rows = []
    for restart in RESTARTS:
        rows.extend(
            [
                {"method": "DADA-CD reference model", "restart": restart, "gamma": np.nan},
                {"method": "DADA optimistic uniform", "restart": restart, "gamma": np.nan},
                {"method": "Multiplex-RWR STRING+pathway", "restart": restart, "gamma": np.nan},
                {"method": "Multiplex-RWR three-view", "restart": restart, "gamma": np.nan},
            ]
        )
        for gamma in GAMMAS:
            rows.append({"method": "Degree-penalized RWR", "restart": restart, "gamma": gamma})
    return rows


def score_for_config(parts: dict[str, np.ndarray], config: dict, weighted_degree: np.ndarray) -> np.ndarray:
    method = config["method"]
    if method == "DADA-CD reference model":
        return parts["dada_cd"]
    if method == "DADA optimistic uniform":
        return parts["dada_optimistic"]
    if method == "Multiplex-RWR STRING+pathway":
        return parts["multiplex_sp"]
    if method == "Multiplex-RWR three-view":
        return parts["multiplex_all"]
    if method == "Degree-penalized RWR":
        return parts["raw"] / np.power(weighted_degree + 1.0, float(config["gamma"]))
    raise KeyError(method)


def historical_tuning(
    labels: np.ndarray,
    transitions: list,
    string_degree: np.ndarray,
    weighted_degree: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    configs = candidate_configs()
    rows = []
    cache: dict = {}
    for split_index, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        seeds = train[labels[train] == 1]
        per_restart = {
            restart: build_method_scores(transitions, string_degree, weighted_degree, seeds, restart, cache)
            for restart in RESTARTS
        }
        for config in configs:
            score = score_for_config(per_restart[config["restart"]], config, weighted_degree)[test]
            _, auprc = safe_metrics(labels[test], score)
            top = np.argsort(-score, kind="mergesort")[: min(100, len(test))]
            rows.append(
                {
                    "repeat": split_index // 5,
                    "fold": split_index % 5,
                    **config,
                    "AUPRC": auprc,
                    "Precision_at_100": float(labels[test][top].mean()),
                    "score_degree_spearman": float(spearmanr(score, string_degree[test]).statistic),
                }
            )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["method", "restart", "gamma"], dropna=False, as_index=False)
        .agg(
            mean_AUPRC=("AUPRC", "mean"),
            sd_AUPRC=("AUPRC", "std"),
            mean_Precision_at_100=("Precision_at_100", "mean"),
            mean_abs_score_degree_spearman=("score_degree_spearman", lambda x: float(np.mean(np.abs(x)))),
        )
        .sort_values(["method", "mean_AUPRC"], ascending=[True, False])
    )
    selected = {}
    for method, block in summary.groupby("method", sort=False):
        row = block.sort_values(
            ["mean_AUPRC", "mean_abs_score_degree_spearman"], ascending=[False, True]
        ).iloc[0]
        selected[method] = {
            "restart": float(row.restart),
            "gamma": None if pd.isna(row.gamma) else float(row.gamma),
            "historical_mean_AUPRC": float(row.mean_AUPRC),
            "historical_mean_Precision_at_100": float(row.mean_Precision_at_100),
        }
    return summary, selected


def dcpi_performance_score(core, transitions, bins, seeds, size: int, config: dict) -> np.ndarray:
    cfg = core.Config(
        float(config["restart"]),
        float(config["blend"]),
        float(config["weight_string"]),
        float(config["weight_celltype"]),
        float(config["weight_pathway"]),
    )
    return core.score_config(core.components(transitions, bins, seeds, size, cfg.restart), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Degree-aware and multiplex-RWR cross-release baselines.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--cancers", nargs="+", choices=CANCERS, default=CANCERS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    root = args.package_root
    core = load_module(root / "code" / "25_degree_conditioned_pathway_rwr.py", "dcpi_primary")
    data = torch.load(root / "inputs" / "GNN_Dataset_Raw.pt", map_location="cpu", weights_only=False)
    mapping, symbols = read_mapping(root / "inputs" / "node_mapping.json")
    size = len(symbols)
    view_specs = [
        ("edge_index_ppi", "edge_weight_ppi"),
        ("edge_index_coexp", "edge_weight_coexp"),
        ("edge_index_pathway", "edge_weight_pathway"),
    ]
    transitions = []
    degrees = []
    for edge_name, weight_name in view_specs:
        edge = getattr(data, edge_name).cpu().numpy()
        weight = getattr(data, weight_name).cpu().numpy().astype(float)
        transitions.append(transition(edge, weight, size))
        degrees.append(np.bincount(edge[0], minlength=size).astype(float))
    string_edge = data.edge_index_ppi.cpu().numpy()
    string_weight = data.edge_weight_ppi.cpu().numpy().astype(float)
    weighted_degree = np.bincount(string_edge[0], weights=string_weight, minlength=size).astype(float)
    bins = [degree_bins(x) for x in degrees]
    selected_dcpi = json.loads(
        (root / "results" / "21_constrained_dcpi_rwr" / "DCPI_RWR_selected_configs.json").read_text(encoding="utf-8")
    )

    all_tuning = []
    all_metrics = []
    all_topk = []
    all_scores = []
    all_bootstrap = []
    selections = {}

    cancers = list(args.cancers)
    for cancer_index, cancer in enumerate(cancers):
        historical = read_binary_labels(root / "results" / "01_labels" / cancer / "historical_26.03.csv", mapping, size)
        current = read_binary_labels(root / "results" / "01_labels" / cancer / "future_26.06.csv", mapping, size)
        candidate_nodes = np.flatnonzero(historical == 0)
        incident_full = ((current == 1) & (historical == 0)).astype(np.int8)
        labels = incident_full[candidate_nodes]
        tuning, selected = historical_tuning(historical, transitions, degrees[0], weighted_degree)
        tuning.insert(0, "cancer", cancer)
        all_tuning.append(tuning)
        selections[cancer] = selected

        cache: dict = {}
        seeds = np.flatnonzero(historical == 1)
        raw_reference = rwr(transitions[0], seeds, size, 0.30)
        score_map = {"RWR-STRING": raw_reference}
        for method, config in selected.items():
            parts = build_method_scores(
                transitions, degrees[0], weighted_degree, seeds, float(config["restart"]), cache
            )
            score_map[method] = score_for_config(parts, {"method": method, **config}, weighted_degree)
        score_map["DCPI-RWR performance"] = dcpi_performance_score(
            core, transitions, bins, seeds, size, selected_dcpi[cancer]["performance"]
        )

        for method, full_score in score_map.items():
            scores = full_score[candidate_nodes]
            roc, auprc = safe_metrics(labels, scores)
            all_metrics.append(
                {
                    "cancer": cancer,
                    "method": method,
                    "historical_positives": int(historical.sum()),
                    "incident_positives": int(labels.sum()),
                    "candidate_universe": int(len(labels)),
                    "ROC_AUC": roc,
                    "AUPRC": auprc,
                    "score_degree_spearman": float(spearmanr(scores, degrees[0][candidate_nodes]).statistic),
                }
            )
            all_topk.extend(
                topk_rows(cancer, method, candidate_nodes, labels, scores, degrees[0])
            )
            all_scores.extend(
                {
                    "cancer": cancer,
                    "method": method,
                    "node_id": int(node),
                    "Gene": symbols[int(node)],
                    "score": float(value),
                    "incident_26.06": int(incident_full[int(node)]),
                    "STRING_degree": float(degrees[0][int(node)]),
                }
                for node, value in zip(candidate_nodes, scores)
            )

        reference = score_map["RWR-STRING"][candidate_nodes]
        alternatives = {k: v[candidate_nodes] for k, v in score_map.items() if k != "RWR-STRING"}
        boot = core.stratified_bootstrap(
            labels,
            reference,
            alternatives,
            args.bootstrap,
            SEED + 3100 + cancer_index,
        )
        boot.insert(0, "cancer", cancer)
        all_bootstrap.append(boot)
        print(f"{cancer}: degree-aware and multiplex baselines complete", flush=True)

    tuning_frame = pd.concat(all_tuning, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    topk = pd.DataFrame(all_topk)
    scores = pd.DataFrame(all_scores)
    bootstrap = pd.concat(all_bootstrap, ignore_index=True)
    bootstrap["p_holm_within_cancer"] = bootstrap.groupby("cancer")["p_two_sided"].transform(
        lambda x: holm_adjust(x.to_numpy(dtype=float))
    )

    tuning_frame.to_csv(args.output / "historical_degree_aware_tuning.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output / "degree_aware_cross_release_metrics.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(args.output / "degree_aware_topk_and_hubs.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(args.output / "degree_aware_full_scores.csv.gz", index=False, compression="gzip")
    bootstrap.to_csv(args.output / "degree_aware_gene_bootstrap.csv", index=False, encoding="utf-8-sig")
    write_json(args.output / "degree_aware_selected_configs.json", selections)
    write_json(
        args.output / "degree_aware_manifest.json",
        {
            "primary_question": "Do established degree-aware or multiplex diffusion baselines outperform raw STRING-RWR or DCPI-RWR under the same cross-release endpoint?",
            "DADA_boundary": "DADA-CD is a documented candidate-degree reference-model reimplementation using 200 nearest weighted-degree genes; it is not a bit-for-bit execution of the historical MATLAB package.",
            "multiplex_definition": "RWR on a convex mixture of column-normalized layer transitions; restart selected using 26.03 cross-validation only.",
            "selection_labels": "Open Targets 26.03 only",
            "evaluation_labels": "incident Open Targets 26.06 labels among historical-unlabelled genes",
            "bootstrap_replicates": args.bootstrap,
        },
    )

    publication_style()
    import matplotlib.pyplot as plt

    order = [
        "RWR-STRING",
        "DADA-CD reference model",
        "DADA optimistic uniform",
        "Degree-penalized RWR",
        "Multiplex-RWR STRING+pathway",
        "Multiplex-RWR three-view",
        "DCPI-RWR performance",
    ]
    colors = ["#0072B2", "#56B4E9", "#CC79A7", "#999999", "#E69F00", "#F0E442", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"wspace": 0.36})
    for method, color in zip(order, colors):
        block = metrics[metrics.method == method].set_index("cancer").reindex(cancers)
        axes[0].plot(cancers, block.AUPRC, marker="o", ms=3.5, lw=1.2, color=color, label=method)
        block100 = topk[(topk.method == method) & (topk.K == 100)].set_index("cancer").reindex(cancers)
        axes[1].plot(cancers, block100.hub_fraction, marker="o", ms=3.5, lw=1.2, color=color, label=method)
    axes[0].set_ylabel("Cross-release incident-gene AUPRC")
    axes[1].set_ylabel("Top-100 STRING hub fraction")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.32)
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.savefig(args.output / "degree_aware_baseline_comparison.pdf", bbox_inches="tight")
    fig.savefig(args.output / "degree_aware_baseline_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

