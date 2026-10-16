from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from q2_upgrade_common import SEED, fixed_rank_by_node, publication_style, rank_average_precision, write_json


FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def main() -> None:
    parser = argparse.ArgumentParser(description="Positive-unlabelled label-ascertainment sensitivity.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.scores)
    max_node = int(frame.node_id.max()) + 1
    methods = list(dict.fromkeys(frame.method.astype(str)))
    raw_rows = []
    rng = np.random.default_rng(SEED + 3300)

    for cancer, cancer_frame in frame.groupby("cancer", sort=False):
        reference = cancer_frame[cancer_frame.method == methods[0]][
            ["node_id", "incident_26.06", "STRING_degree"]
        ].sort_values("node_id")
        candidates = reference.node_id.to_numpy(dtype=int)
        positive_nodes = reference.loc[reference["incident_26.06"] == 1, "node_id"].to_numpy(dtype=int)
        degree_by_node = np.zeros(max_node, dtype=float)
        degree_by_node[reference.node_id.to_numpy(dtype=int)] = reference.STRING_degree.to_numpy(dtype=float)
        rank_maps = {}
        for method in methods:
            block = cancer_frame[cancer_frame.method == method].set_index("node_id").reindex(candidates)
            rank_maps[method] = fixed_rank_by_node(candidates, block.score.to_numpy(dtype=float), max_node)

        for mechanism in ("SCAR uniform", "degree-biased ascertainment"):
            for fraction in FRACTIONS:
                draws = 1 if fraction == 1.0 else args.replicates
                sample_size = max(1, int(round(fraction * len(positive_nodes))))
                if mechanism == "degree-biased ascertainment":
                    weights = np.sqrt(degree_by_node[positive_nodes] + 1.0)
                    weights = weights / weights.sum()
                else:
                    weights = None
                for replicate in range(draws):
                    if fraction == 1.0:
                        selected = positive_nodes
                    else:
                        selected = rng.choice(positive_nodes, size=sample_size, replace=False, p=weights)
                    for method in methods:
                        ranks = rank_maps[method]
                        raw_rows.append(
                            {
                                "cancer": cancer,
                                "mechanism": mechanism,
                                "retained_positive_fraction": fraction,
                                "replicate": replicate,
                                "method": method,
                                "observed_positive_count": len(selected),
                                "observed_AUPRC": rank_average_precision(selected, ranks),
                                "observed_top100_hits": int(np.count_nonzero(ranks[selected] <= 100)),
                            }
                        )
        print(f"{cancer}: PU ascertainment stress test complete", flush=True)

    raw = pd.DataFrame(raw_rows)
    reference_name = "RWR-STRING"
    wide = raw.pivot_table(
        index=["cancer", "mechanism", "retained_positive_fraction", "replicate"],
        columns="method",
        values="observed_AUPRC",
    )
    delta_rows = []
    for method in methods:
        if method == reference_name:
            continue
        temp = (wide[method] - wide[reference_name]).rename("delta").reset_index()
        for keys, block in temp.groupby(["cancer", "mechanism", "retained_positive_fraction"], sort=False):
            values = block.delta.to_numpy(dtype=float)
            delta_rows.append(
                {
                    "cancer": keys[0],
                    "mechanism": keys[1],
                    "retained_positive_fraction": keys[2],
                    "method": method,
                    "reference": reference_name,
                    "median_delta_AUPRC": float(np.median(values)),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "probability_delta_gt_zero": float(np.mean(values > 0)),
                    "replicates": len(values),
                }
            )
    deltas = pd.DataFrame(delta_rows)
    summary = (
        raw.groupby(["cancer", "mechanism", "retained_positive_fraction", "method"], as_index=False)
        .agg(
            median_observed_AUPRC=("observed_AUPRC", "median"),
            auprc_ci_lower=("observed_AUPRC", lambda x: float(np.quantile(x, 0.025))),
            auprc_ci_upper=("observed_AUPRC", lambda x: float(np.quantile(x, 0.975))),
            median_top100_hits=("observed_top100_hits", "median"),
        )
    )
    raw.to_csv(args.output / "pu_ascertainment_replicates.csv.gz", index=False, compression="gzip")
    summary.to_csv(args.output / "pu_ascertainment_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(args.output / "pu_ascertainment_delta_vs_rwr.csv", index=False, encoding="utf-8-sig")
    write_json(
        args.output / "pu_ascertainment_manifest.json",
        {
            "purpose": "Stress-test method ordering when only a subset of incident positives would have been observed.",
            "SCAR": "Incident positives are retained uniformly without replacement.",
            "degree_biased": "Incident positives are retained without replacement with probability proportional to sqrt(STRING degree + 1).",
            "boundary": "Predictions are fixed. This analysis does not identify latent true positives and is not a causal or prevalence-corrected PU estimator.",
            "replicates_per_incomplete_fraction": args.replicates,
            "random_seed": SEED + 3300,
        },
    )

    publication_style()
    import matplotlib.pyplot as plt

    focus = summary[summary.method.isin(["RWR-STRING", "DCPI-RWR performance", "Multiplex-RWR three-view"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.18})
    colors = {"RWR-STRING": "#0072B2", "DCPI-RWR performance": "#D55E00", "Multiplex-RWR three-view": "#009E73"}
    for ax, mechanism in zip(axes, ("SCAR uniform", "degree-biased ascertainment")):
        part = focus[focus.mechanism == mechanism]
        for method, color in colors.items():
            block = (
                part[part.method == method]
                .groupby("retained_positive_fraction", as_index=False)
                .agg(median_observed_AUPRC=("median_observed_AUPRC", "mean"))
            )
            ax.plot(block.retained_positive_fraction, block.median_observed_AUPRC, marker="o", lw=1.3, ms=3.5, color=color, label=method)
        ax.set_title(mechanism)
        ax.set_xlabel("Observed fraction of incident positives")
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    axes[0].set_ylabel("Mean cancer-level observed AUPRC")
    axes[0].legend(frameon=False)
    fig.savefig(args.output / "pu_label_ascertainment_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(args.output / "pu_label_ascertainment_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
