from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from upgrade2_common import metrics, read_labels, read_mapping, topk_summary, transition, weighted_rwr, write_json


GROUPS = {
    "Epithelial": ["Epithelial"],
    "Fibrovascular": ["Fibroblasts", "Endothelial"],
    "Myeloid_DC": ["pDC", "cDC2", "Monocyte", "M2"],
    "T_cell": ["CD4Tn", "CD8Tem", "CD8Tcm", "CD8Tex"],
    "B_Plasma": ["B", "Plasma"],
    "Mast": ["Mast"],
}


def group_transition(data, indices, size):
    x = data.x.cpu().numpy().astype(float)
    activity = x[:, indices].mean(axis=1)
    pct = rankdata(activity, method="average") / len(activity)
    edge = data.edge_index_ppi.cpu().numpy()
    base = data.edge_weight_ppi.cpu().numpy().astype(float)
    factor = 0.25 + 0.75 * np.sqrt(pct[edge[0]] * pct[edge[1]])
    return transition(edge, base * factor, size), activity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.package_root
    data = torch.load(root / "inputs" / "GNN_Dataset_Raw.pt", map_location="cpu", weights_only=False)
    mapping, symbols = read_mapping(root / "inputs" / "node_mapping.json")
    size = len(symbols)
    names = [str(x) for x in data.cell_types]
    edge = data.edge_index_ppi.cpu().numpy()
    base_weight = data.edge_weight_ppi.cpu().numpy().astype(float)
    base_matrix = transition(edge, base_weight, size)
    degree = np.bincount(edge[0], minlength=size).astype(float)
    hub_cutoff = float(np.quantile(degree, 0.95))
    historical = read_labels(root / "results" / "01_labels" / "LUAD" / "historical_26.03.csv", mapping, size)
    future = read_labels(root / "results" / "01_labels" / "LUAD" / "future_26.06.csv", mapping, size)
    incident = ((future == 1) & (historical == 0)).astype(np.int8)
    candidates = np.flatnonzero(historical == 0)
    seeds = np.flatnonzero(historical == 1)
    rows = []
    scores = {}
    activities = {}
    scores["STRING unweighted"] = weighted_rwr(base_matrix, seeds, size, 0.30)
    for group, members in GROUPS.items():
        idx = [names.index(x) for x in members if x in names]
        matrix, activity = group_transition(data, idx, size)
        scores[f"STRING context: {group}"] = weighted_rwr(matrix, seeds, size, 0.30)
        activities[group] = activity
    y = incident[candidates]
    for method, full in scores.items():
        s = full[candidates]
        roc, auprc = metrics(y, s)
        rows.append(
            {
                "method": method,
                "ROC_AUC": roc,
                "AUPRC": auprc,
                "score_degree_spearman": float(spearmanr(s, degree[candidates]).statistic),
                **topk_summary(y, s, degree[candidates], hub_cutoff=hub_cutoff),
            }
        )
    pd.DataFrame(rows).to_csv(args.output / "LUAD_context_layer_ablation.csv", index=False, encoding="utf-8-sig")
    activity_frame = pd.DataFrame({"Gene": symbols, **activities})
    activity_frame.to_csv(args.output / "LUAD_cell_group_activity.csv.gz", index=False, compression="gzip")
    write_json(
        args.output / "context_layer_manifest.json",
        {
            "scope": "LUAD only",
            "boundary": "The graph is STRING reweighted by packaged TISCH cell-group activity. It is not a de novo cell-type regulatory network.",
            "floor": 0.25,
            "evaluation": "Open Targets 26.06 incident labels among 26.03-unlabelled genes",
        },
    )


if __name__ == "__main__":
    main()

