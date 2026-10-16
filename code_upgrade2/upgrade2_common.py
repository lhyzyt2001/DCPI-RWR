from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


CANCERS = ("LUAD", "BRCA", "CRC", "PRAD", "MEL", "OV")
SEED = 20261014


def read_mapping(path: Path) -> tuple[dict[str, int], list[str]]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    symbols = [""] * len(mapping)
    for symbol, node in mapping.items():
        symbols[int(node)] = str(symbol)
    return {str(k): int(v) for k, v in mapping.items()}, symbols


def read_labels(path: Path, mapping: dict[str, int], size: int) -> np.ndarray:
    frame = pd.read_csv(path)
    column = next(c for c in frame.columns if c.lower() in {"symbol", "gene", "gene_symbol"})
    labels = np.zeros(size, dtype=np.int8)
    for symbol in frame[column].astype(str):
        node = mapping.get(symbol)
        if node is not None:
            labels[node] = 1
    return labels


def transition(edge: np.ndarray, weight: np.ndarray, size: int) -> sp.csc_matrix:
    adjacency = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(size, size)).tocsc()
    sums = np.asarray(adjacency.sum(axis=0)).ravel()
    sums[sums <= 0] = 1.0
    return (adjacency @ sp.diags(1.0 / sums)).tocsc()


def weighted_rwr(
    matrix: sp.csc_matrix,
    seed_nodes: np.ndarray,
    size: int,
    restart: float,
    seed_weights: np.ndarray | None = None,
    max_iter: int = 250,
    tol: float = 1e-10,
) -> np.ndarray:
    initial = np.zeros(size, dtype=float)
    if len(seed_nodes):
        if seed_weights is None:
            weights = np.ones(len(seed_nodes), dtype=float)
        else:
            weights = np.asarray(seed_weights, dtype=float)
        weights = np.maximum(weights, 1e-12)
        weights /= weights.sum()
        initial[np.asarray(seed_nodes, dtype=int)] = weights
    score = initial.copy()
    for _ in range(max_iter):
        updated = (1.0 - restart) * matrix.dot(score) + restart * initial
        if np.linalg.norm(updated - score, 1) < tol:
            return np.asarray(updated).ravel()
        score = updated
    return np.asarray(score).ravel()


def supra_rwr(
    matrices: list[sp.csc_matrix],
    seed_nodes: np.ndarray,
    size: int,
    restart: float,
    layer_weights: np.ndarray,
    switch: float,
    seed_weights: np.ndarray | None = None,
    max_iter: int = 180,
    tol: float = 1e-9,
) -> np.ndarray:
    """RWR on an implicit supra-transition matrix.

    A walker follows the current layer with probability 1-switch and reselects a
    layer according to layer_weights with probability switch. Restart mass uses
    the same prespecified layer distribution. The operator is column stochastic.
    """

    weights = np.asarray(layer_weights, dtype=float)
    weights = weights / weights.sum()
    base = np.zeros(size, dtype=float)
    if len(seed_nodes):
        if seed_weights is None:
            sw = np.ones(len(seed_nodes), dtype=float)
        else:
            sw = np.maximum(np.asarray(seed_weights, dtype=float), 1e-12)
        sw /= sw.sum()
        base[np.asarray(seed_nodes, dtype=int)] = sw
    initial = weights[:, None] * base[None, :]
    score = initial.copy()
    for _ in range(max_iter):
        total = score.sum(axis=0)
        propagated = np.vstack([matrix.dot(score[i]) for i, matrix in enumerate(matrices)])
        moved = (1.0 - switch) * propagated + switch * weights[:, None] * total[None, :]
        updated = (1.0 - restart) * moved + restart * initial
        if np.abs(updated - score).sum() < tol:
            score = updated
            break
        score = updated
    return np.asarray(score.sum(axis=0)).ravel()


def evidence_weights(path: Path, mapping: dict[str, int], size: int) -> np.ndarray:
    frame = pd.read_csv(path)
    output = np.ones(size, dtype=float)
    if frame.empty:
        return output
    phase_map = {
        "PHASE_1": 1.0,
        "PHASE_1_2": 1.5,
        "PHASE_2": 2.0,
        "PHASE_2_3": 2.5,
        "PHASE_3": 3.0,
        "PHASE_4": 4.0,
    }
    frame = frame.copy()
    frame["phase_value"] = frame.get("clinicalStage", "").map(phase_map).fillna(1.0)
    rows = []
    for symbol, block in frame.groupby("symbol", sort=False):
        trials = block.get("clinicalReportId", pd.Series(dtype=object)).nunique()
        drugs = block.get("drugId", pd.Series(dtype=object)).nunique()
        strength = (float(block.phase_value.max()) / 4.0) ** 0.5
        diversity = (np.log1p(max(trials, 1)) * np.log1p(max(drugs, 1))) ** 0.25
        rows.append((str(symbol), strength * diversity))
    if rows:
        values = np.array([v for _, v in rows], dtype=float)
        scale = np.median(values[values > 0]) if np.any(values > 0) else 1.0
        for symbol, value in rows:
            node = mapping.get(symbol)
            if node is not None:
                output[node] = float(np.clip(value / scale, 0.25, 4.0))
    return output


def context_transition(data, size: int) -> tuple[sp.csc_matrix, pd.DataFrame]:
    """Construct a LUAD-context STRING view from packaged TISCH profiles.

    This is an activity-reweighted STRING network, not a de novo regulatory
    network. A 0.25 floor retains edges for genes with low measured activity.
    """

    x = data.x.cpu().numpy().astype(float)
    names = [str(x) for x in data.cell_types]
    groups = {
        "Epithelial": ["Epithelial"],
        "Fibrovascular": ["Fibroblasts", "Endothelial"],
        "Myeloid_DC": ["pDC", "cDC2", "Monocyte", "M2"],
        "T_cell": ["CD4Tn", "CD8Tem", "CD8Tcm", "CD8Tex"],
        "B_Plasma": ["B", "Plasma"],
        "Mast": ["Mast"],
    }
    profiles = []
    rows = []
    for group, members in groups.items():
        idx = [names.index(member) for member in members if member in names]
        activity = x[:, idx].mean(axis=1)
        pct = rankdata(activity, method="average") / len(activity)
        profiles.append(pct)
        rows.append({"group": group, "members": ";".join(members), "mean_activity": float(activity.mean())})
    context = np.mean(np.vstack(profiles), axis=0)
    edge = data.edge_index_ppi.cpu().numpy()
    base = data.edge_weight_ppi.cpu().numpy().astype(float)
    factor = 0.25 + 0.75 * np.sqrt(context[edge[0]] * context[edge[1]])
    return transition(edge, base * factor, size), pd.DataFrame(rows)


def metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def topk_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    degrees: np.ndarray,
    k: int = 100,
    hub_cutoff: float | None = None,
) -> dict:
    order = np.argsort(-scores, kind="mergesort")[: min(k, len(scores))]
    cutoff = float(np.quantile(degrees, 0.95)) if hub_cutoff is None else float(hub_cutoff)
    return {
        "top100_hits": int(labels[order].sum()),
        "top100_precision": float(labels[order].mean()),
        "top100_hub_fraction": float((degrees[order] >= cutoff).mean()),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

