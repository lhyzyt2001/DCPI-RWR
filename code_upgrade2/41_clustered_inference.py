from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    adaptive = pd.read_csv(args.results / "01_adaptive_supra" / "adaptive_supra_full_scores.csv.gz")
    pu = pd.read_csv(args.results / "03_propensity_pu" / "propensity_pu_scores.csv.gz")
    pu["method"] = "Degree-propensity-weighted bagged PU stack"
    columns = ["cancer", "method", "node_id", "Gene", "score", "incident_26.06", "STRING_degree"]
    combined = pd.concat([adaptive[columns], pu[columns]], ignore_index=True)
    score_path = args.output / "fivepoint_full_scores.csv.gz"
    combined.to_csv(score_path, index=False, compression="gzip")
    script = args.package_root / "code_upgrade" / "34_clustered_top100_inference.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.package_root / "code_upgrade")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--package-root",
            str(args.package_root),
            "--scores",
            str(score_path),
            "--output",
            str(args.output),
            "--replicates",
            str(args.replicates),
            "--null-permutations",
            "0",
        ],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()

