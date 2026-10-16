from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from upgrade2_common import (
    CANCERS,
    SEED,
    context_transition,
    evidence_weights,
    metrics,
    percentile,
    read_labels,
    read_mapping,
    supra_rwr,
    topk_summary,
    transition,
    weighted_rwr,
    write_json,
)


PROFILES = (
    (1.00, 0.00, 0.00),
    (0.80, 0.20, 0.00),
    (0.80, 0.00, 0.20),
    (0.60, 0.20, 0.20),
    (0.60, 0.40, 0.00),
    (0.60, 0.00, 0.40),
    (0.34, 0.33, 0.33),
)
RESTARTS = (0.15, 0.30, 0.50)
SWITCHES = (0.10, 0.25)
SEED_MODES = ("uniform", "evidence_weighted")


def run_score(matrices, seeds, size, config, evidence):
    sw = None if config["seed_mode"] == "uniform" else evidence[seeds]
    return supra_rwr(
        matrices,
        seeds,
        size,
        config["restart"],
        np.asarray(config["layer_weights"]),
        config["switch"],
        sw,
    )


def configs():
    for restart in RESTARTS:
        for switch in SWITCHES:
            for weights in PROFILES:
                for seed_mode in SEED_MODES:
                    yield {
                        "restart": restart,
                        "switch": switch,
                        "layer_weights": weights,
                        "seed_mode": seed_mode,
                    }


def key(config):
    w = config["layer_weights"]
    return f"r{config['restart']:.2f}_s{config['switch']:.2f}_w{w[0]:.2f}-{w[1]:.2f}-{w[2]:.2f}_{config['seed_mode']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancers", nargs="+", choices=CANCERS, default=CANCERS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.package_root
    data = torch.load(root / "inputs" / "GNN_Dataset_Raw.pt", map_location="cpu", weights_only=False)
    mapping, symbols = read_mapping(root / "inputs" / "node_mapping.json")
    size = len(symbols)
    specs = (
        ("edge_index_ppi", "edge_weight_ppi"),
        ("edge_index_coexp", "edge_weight_coexp"),
        ("edge_index_pathway", "edge_weight_pathway"),
    )
    matrices = []
    for edge_name, weight_name in specs:
        edge = getattr(data, edge_name).cpu().numpy()
        weight = getattr(data, weight_name).cpu().numpy().astype(float)
        matrices.append(transition(edge, weight, size))
    context_matrix, context_meta = context_transition(data, size)
    context_meta.to_csv(args.output / "LUAD_context_groups.csv", index=False, encoding="utf-8-sig")
    string_edge = data.edge_index_ppi.cpu().numpy()
    degree = np.bincount(string_edge[0], minlength=size).astype(float)
    hub_cutoff = float(np.quantile(degree, 0.95))

    tuning_rows = []
    selection = {}
    score_rows = []
    metric_rows = []
    oof_rows = []
    for cancer_index, cancer in enumerate(args.cancers):
        historical = read_labels(root / "results" / "01_labels" / cancer / "historical_26.03.csv", mapping, size)
        future = read_labels(root / "results" / "01_labels" / cancer / "future_26.06.csv", mapping, size)
        incident = ((future == 1) & (historical == 0)).astype(np.int8)
        evidence = evidence_weights(
            root / "results" / "01_labels" / cancer / "26.03_disease_specific_evidence.csv.gz", mapping, size
        )
        cancer_matrices = [matrices[0], context_matrix if cancer == "LUAD" else matrices[1], matrices[2]]
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + cancer_index)
        cfgs = list(configs())
        fold_records = []
        for fold, (train, test) in enumerate(splitter.split(np.zeros(size), historical)):
            seeds = train[historical[train] == 1]
            for config in cfgs:
                score = run_score(cancer_matrices, seeds, size, config, evidence)
                roc, auprc = metrics(historical[test], score[test])
                degree_rho = float(spearmanr(score[test], degree[test]).statistic)
                top = topk_summary(historical[test], score[test], degree[test], hub_cutoff=hub_cutoff)
                fold_records.append(
                    {
                        "cancer": cancer,
                        "fold": fold,
                        "config_id": key(config),
                        "ROC_AUC": roc,
                        "AUPRC": auprc,
                        "abs_degree_rho": abs(degree_rho),
                        **top,
                    }
                )
        folds = pd.DataFrame(fold_records)
        tuning_rows.append(folds)
        summary = folds.groupby("config_id", as_index=False).agg(
            mean_AUPRC=("AUPRC", "mean"),
            mean_ROC_AUC=("ROC_AUC", "mean"),
            mean_abs_degree_rho=("abs_degree_rho", "mean"),
            mean_top100_hub_fraction=("top100_hub_fraction", "mean"),
        )
        perf_id = summary.sort_values(
            ["mean_AUPRC", "mean_abs_degree_rho"], ascending=[False, True]
        ).iloc[0].config_id
        cutoff = 0.95 * float(summary.mean_AUPRC.max())
        eligible = summary[summary.mean_AUPRC >= cutoff].copy()
        bias_id = eligible.sort_values(
            ["mean_top100_hub_fraction", "mean_abs_degree_rho", "mean_AUPRC"],
            ascending=[True, True, False],
        ).iloc[0].config_id
        by_id = {key(c): c for c in cfgs}
        selection[cancer] = {
            "performance": {**by_id[perf_id], "layer_weights": list(by_id[perf_id]["layer_weights"])},
            "bias_aware_95pct": {**by_id[bias_id], "layer_weights": list(by_id[bias_id]["layer_weights"])},
        }

        # OOF diffusion features for PU stacking. Each positive is excluded from its own restart set.
        oof = {"raw_rwr": np.zeros(size), "adaptive_performance": np.zeros(size), "adaptive_bias_aware": np.zeros(size)}
        for fold, (train, test) in enumerate(splitter.split(np.zeros(size), historical)):
            seeds = train[historical[train] == 1]
            oof["raw_rwr"][test] = weighted_rwr(matrices[0], seeds, size, 0.30)[test]
            oof["adaptive_performance"][test] = run_score(cancer_matrices, seeds, size, by_id[perf_id], evidence)[test]
            oof["adaptive_bias_aware"][test] = run_score(cancer_matrices, seeds, size, by_id[bias_id], evidence)[test]
        for node in range(size):
            oof_rows.append(
                {
                    "cancer": cancer,
                    "node_id": node,
                    "Gene": symbols[node],
                    "historical_label": int(historical[node]),
                    **{name: float(values[node]) for name, values in oof.items()},
                    "STRING_degree": float(degree[node]),
                }
            )

        seeds = np.flatnonzero(historical == 1)
        methods = {
            "RWR-STRING": weighted_rwr(matrices[0], seeds, size, 0.30),
            "Adaptive-supra performance": run_score(cancer_matrices, seeds, size, by_id[perf_id], evidence),
            "Adaptive-supra bias-aware": run_score(cancer_matrices, seeds, size, by_id[bias_id], evidence),
        }
        # Isolate the value of confidence weighting at the selected topology parameters.
        weighted_cfg = dict(by_id[perf_id])
        weighted_cfg["seed_mode"] = "evidence_weighted"
        methods["Adaptive-supra evidence-weighted"] = run_score(cancer_matrices, seeds, size, weighted_cfg, evidence)
        candidates = np.flatnonzero(historical == 0)
        y = incident[candidates]
        for method, full_score in methods.items():
            score = full_score[candidates]
            roc, auprc = metrics(y, score)
            rho = float(spearmanr(score, degree[candidates]).statistic)
            top = topk_summary(y, score, degree[candidates], hub_cutoff=hub_cutoff)
            metric_rows.append(
                {
                    "cancer": cancer,
                    "method": method,
                    "ROC_AUC": roc,
                    "AUPRC": auprc,
                    "score_degree_spearman": rho,
                    "incident_positives": int(y.sum()),
                    **top,
                }
            )
            for node, value in zip(candidates, score):
                score_rows.append(
                    {
                        "cancer": cancer,
                        "method": method,
                        "node_id": int(node),
                        "Gene": symbols[int(node)],
                        "score": float(value),
                        "incident_26.06": int(incident[int(node)]),
                        "STRING_degree": float(degree[int(node)]),
                    }
                )
        print(f"{cancer}: adaptive supra-RWR complete", flush=True)

    pd.concat(tuning_rows, ignore_index=True).to_csv(
        args.output / "adaptive_supra_historical_cv.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(metric_rows).to_csv(args.output / "adaptive_supra_cross_release_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(score_rows).to_csv(args.output / "adaptive_supra_full_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(oof_rows).to_csv(args.output / "adaptive_supra_oof_features.csv.gz", index=False, compression="gzip")
    write_json(args.output / "adaptive_supra_selected_configs.json", selection)
    write_json(
        args.output / "adaptive_supra_manifest.json",
        {
            "selection_labels": "Open Targets 26.03 only",
            "evaluation_labels": "Open Targets 26.06 incident labels among historical-unlabelled genes",
            "performance_selection": "Highest 3-fold historical AUPRC; lower absolute degree correlation breaks ties.",
            "bias_selection": "Lowest top-100 hub fraction among configurations retaining >=95% of maximum historical AUPRC.",
            "LUAD_cell_layer": "Activity-reweighted STRING using packaged 15-cell-type TISCH profiles.",
            "other_cancers_cell_layer": "Original packaged generic cell-type similarity graph; no cancer-specific single-cell data were available.",
        },
    )


if __name__ == "__main__":
    main()

