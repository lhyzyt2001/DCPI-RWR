from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 42
CANCERS = ("LUAD", "BRCA", "CRC", "PRAD", "MEL", "OV")


def load_module(path: Path, name: str = "dcpi_core"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_mapping(path: Path) -> tuple[dict[str, int], list[str]]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    symbols = [""] * len(mapping)
    for symbol, node in mapping.items():
        symbols[int(node)] = symbol
    return {str(k): int(v) for k, v in mapping.items()}, symbols


def read_binary_labels(path: Path, mapping: dict[str, int], size: int) -> np.ndarray:
    frame = pd.read_csv(path)
    labels = np.zeros(size, dtype=np.int8)
    symbol_col = next(c for c in frame.columns if c.lower() in {"symbol", "gene", "gene_symbol"})
    for symbol in frame[symbol_col].astype(str):
        node = mapping.get(symbol)
        if node is not None:
            labels[node] = 1
    return labels


def transition(edge: np.ndarray, weight: np.ndarray, size: int) -> sp.csc_matrix:
    adjacency = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(size, size)).tocsc()
    column_sum = np.asarray(adjacency.sum(axis=0)).ravel()
    column_sum[column_sum == 0] = 1.0
    return adjacency @ sp.diags(1.0 / column_sum)


def rwr(matrix: sp.csc_matrix, seeds: np.ndarray, size: int, restart: float) -> np.ndarray:
    initial = np.zeros(size, dtype=float)
    if len(seeds):
        initial[np.asarray(seeds, dtype=int)] = 1.0 / len(seeds)
    score = initial.copy()
    for _ in range(300):
        updated = (1.0 - restart) * matrix.dot(score) + restart * initial
        if np.linalg.norm(updated - score, 1) < 1e-10:
            return np.asarray(updated).ravel()
        score = updated
    return np.asarray(score).ravel()


def degree_bins(degree: np.ndarray, requested: int = 20) -> np.ndarray:
    try:
        return np.asarray(
            pd.qcut(np.log1p(degree.astype(float)), q=requested, labels=False, duplicates="drop"),
            dtype=int,
        )
    except ValueError:
        return np.zeros(len(degree), dtype=int)


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def conditioned_percentile(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    for bin_id in np.unique(bins):
        selected = np.flatnonzero(bins == bin_id)
        output[selected] = rankdata(values[selected], method="average") / len(selected)
    return output


def optimistic_uniform_score(raw: np.ndarray, adjusted: np.ndarray) -> np.ndarray:
    """DADA optimistic-uniform-style merge of raw and adjusted candidate ranks.

    This is deliberately labelled as DADA-style rather than a bit-for-bit reimplementation
    of the historical MATLAB package. The best rank is primary and the unused rank breaks ties.
    """

    raw_rank = rankdata(-raw, method="average")
    adjusted_rank = rankdata(-adjusted, method="average")
    best = np.minimum(raw_rank, adjusted_rank)
    other = np.maximum(raw_rank, adjusted_rank)
    order = np.lexsort((other, best))
    score = np.empty(len(raw), dtype=float)
    score[order] = np.linspace(1.0, 0.0, len(raw), endpoint=True)
    return score


def average_transition(matrices: list[sp.csc_matrix], weights: list[float]) -> sp.csc_matrix:
    total = matrices[0].copy() * float(weights[0])
    for matrix, weight in zip(matrices[1:], weights[1:]):
        total = total + matrix * float(weight)
    return total.tocsc()


def safe_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def topk_rows(
    cancer: str,
    method: str,
    candidate_nodes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    string_degree: np.ndarray,
    ks: tuple[int, ...] = (20, 50, 100, 200, 500, 1000),
) -> list[dict]:
    order = np.argsort(-scores, kind="mergesort")
    hub_cutoff = float(np.quantile(string_degree, 0.95))
    rows = []
    for k in ks:
        chosen = order[: min(k, len(order))]
        nodes = candidate_nodes[chosen]
        rows.append(
            {
                "cancer": cancer,
                "method": method,
                "K": k,
                "incident_hits": int(labels[chosen].sum()),
                "precision_at_K": float(labels[chosen].mean()),
                "incident_recall": float(labels[chosen].sum() / max(labels.sum(), 1)),
                "hub_fraction": float((string_degree[nodes] >= hub_cutoff).mean()),
                "median_STRING_degree": float(np.median(string_degree[nodes])),
            }
        )
    return rows


def rank_average_precision(selected_positive_nodes: np.ndarray, rank_by_node: np.ndarray) -> float:
    positions = np.sort(rank_by_node[selected_positive_nodes])
    if len(positions) == 0:
        return np.nan
    return float(np.mean(np.arange(1, len(positions) + 1) / positions))


def fixed_rank_by_node(candidate_nodes: np.ndarray, scores: np.ndarray, size: int) -> np.ndarray:
    # Stable node-id tie breaking makes the sensitivity analysis deterministic.
    order = np.lexsort((candidate_nodes, -scores))
    rank = np.full(size, np.iinfo(np.int32).max, dtype=np.int32)
    rank[candidate_nodes[order]] = np.arange(1, len(candidate_nodes) + 1, dtype=np.int32)
    return rank


def holm_adjust(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    output = np.empty(len(values), dtype=float)
    running = 0.0
    for step, index in enumerate(order):
        adjusted = min(1.0, (len(values) - step) * values[index])
        running = max(running, adjusted)
        output[index] = running
    return output


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

