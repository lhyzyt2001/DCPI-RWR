from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from upgrade2_common import CANCERS, SEED, metrics, read_labels, read_mapping, topk_summary, write_json


def design(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(frame["STRING_degree"].to_numpy(float)),
            np.log10(np.maximum(frame["raw_rwr"].to_numpy(float), 1e-15)),
            np.log10(np.maximum(frame["adaptive_performance"].to_numpy(float), 1e-15)),
            np.log10(np.maximum(frame["adaptive_bias_aware"].to_numpy(float), 1e-15)),
        ]
    )


def bagged_pu(train_x, train_y, pred_x, propensity, rng, bags=30):
    positives = np.flatnonzero(train_y == 1)
    unlabeled = np.flatnonzero(train_y == 0)
    # High ascertainment-proxy unlabeled genes are less likely to be hidden merely
    # because they are peripheral. This is a sensitivity heuristic, not an identified SAR estimator.
    probability = np.clip(propensity[unlabeled], 0.02, 0.98)
    probability = probability / probability.sum()
    predictions = []
    for bag in range(bags):
        negative = rng.choice(unlabeled, size=min(len(unlabeled), max(3 * len(positives), 250)), replace=False, p=probability)
        selected = np.concatenate([positives, negative])
        y = np.concatenate([np.ones(len(positives), dtype=int), np.zeros(len(negative), dtype=int)])
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=140,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED + bag,
        )
        pos_weight = np.clip(1.0 / propensity[positives], 1.0, 10.0)
        sample_weight = np.concatenate([pos_weight, np.ones(len(negative))])
        model.fit(train_x[selected], y, sample_weight=sample_weight)
        predictions.append(model.predict_proba(pred_x)[:, 1])
    return np.mean(np.vstack(predictions), axis=0), np.std(np.vstack(predictions), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--adaptive-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping, symbols = read_mapping(args.package_root / "inputs" / "node_mapping.json")
    size = len(symbols)
    oof = pd.read_csv(args.adaptive_results / "adaptive_supra_oof_features.csv.gz")
    full = pd.read_csv(args.adaptive_results / "adaptive_supra_full_scores.csv.gz")
    rng = np.random.default_rng(SEED)
    metric_rows, score_rows, propensity_rows = [], [], []
    for cancer_index, cancer in enumerate(CANCERS):
        train = oof[oof.cancer == cancer].sort_values("node_id").reset_index(drop=True)
        historical = train.historical_label.to_numpy(np.int8)
        # Use the final full-data scores corresponding to the same three features.
        block = full[full.cancer == cancer]
        wide = block.pivot(index="node_id", columns="method", values="score").reindex(np.arange(size))
        pred = train[["cancer", "node_id", "Gene", "historical_label", "STRING_degree"]].copy()
        pred["raw_rwr"] = wide["RWR-STRING"].to_numpy()
        pred["adaptive_performance"] = wide["Adaptive-supra performance"].to_numpy()
        pred["adaptive_bias_aware"] = wide["Adaptive-supra bias-aware"].to_numpy()
        scaler = StandardScaler().fit(design(train))
        x_train = scaler.transform(design(train))
        x_pred = scaler.transform(design(pred))
        propensity_model = LogisticRegression(
            C=0.2, class_weight="balanced", max_iter=1000, random_state=SEED + cancer_index
        )
        propensity_model.fit(x_train[:, [0]], historical)
        propensity = np.clip(propensity_model.predict_proba(x_train[:, [0]])[:, 1], 0.02, 0.98)
        mean, sd = bagged_pu(x_train, historical, x_pred, propensity, rng)
        future = read_labels(args.package_root / "results" / "01_labels" / cancer / "future_26.06.csv", mapping, size)
        incident = ((future == 1) & (historical == 0)).astype(np.int8)
        candidates = np.flatnonzero(historical == 0)
        y = incident[candidates]
        degree = train.STRING_degree.to_numpy(float)
        hub_cutoff = float(np.quantile(degree, 0.95))
        roc, auprc = metrics(y, mean[candidates])
        metric_rows.append(
            {
                "cancer": cancer,
                "method": "Degree-propensity-weighted bagged PU stack",
                "ROC_AUC": roc,
                "AUPRC": auprc,
                "incident_positives": int(y.sum()),
                **topk_summary(y, mean[candidates], degree[candidates], hub_cutoff=hub_cutoff),
            }
        )
        for node in candidates:
            score_rows.append(
                {
                    "cancer": cancer,
                    "node_id": int(node),
                    "Gene": symbols[int(node)],
                    "score": float(mean[node]),
                    "ensemble_sd": float(sd[node]),
                    "incident_26.06": int(incident[node]),
                    "STRING_degree": float(degree[node]),
                }
            )
        propensity_rows.append(
            {
                "cancer": cancer,
                "min": float(propensity.min()),
                "median": float(np.median(propensity)),
                "max": float(propensity.max()),
                "coefficient_log_degree": float(propensity_model.coef_[0, 0]),
            }
        )
        print(f"{cancer}: PU stack complete", flush=True)
    pd.DataFrame(metric_rows).to_csv(args.output / "propensity_pu_cross_release_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(score_rows).to_csv(args.output / "propensity_pu_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(propensity_rows).to_csv(args.output / "label_propensity_proxy_audit.csv", index=False, encoding="utf-8-sig")
    write_json(
        args.output / "propensity_pu_manifest.json",
        {
            "features": ["cross-fitted STRING-RWR", "cross-fitted adaptive supra performance", "cross-fitted adaptive supra bias-aware", "log1p STRING degree"],
            "bags": 30,
            "negative_sampling": "High degree-based historical-label propensity proxy",
            "boundary": "This is a propensity-proxy PU sensitivity model, not an identified SAR-PU estimator. Publication count and true observation probabilities were unavailable.",
        },
    )


if __name__ == "__main__":
    main()

