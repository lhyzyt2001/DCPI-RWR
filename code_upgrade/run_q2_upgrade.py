from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\nRUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent DCPI-RWR four-priority upgrade.")
    parser.add_argument("--upgrade-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--pu-replicates", type=int, default=1000)
    parser.add_argument("--cluster-replicates", type=int, default=10000)
    parser.add_argument("--null-permutations", type=int, default=2000)
    args = parser.parse_args()
    root = args.upgrade_root.resolve()
    code = root / "code_upgrade"
    results = root / "results_upgrade"
    directories = {
        "degree": results / "01_degree_aware_multiplex",
        "boundary": results / "02_cross_release_boundary",
        "pu": results / "03_pu_sensitivity",
        "cluster": results / "04_clustered_inference",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    run([python, str(code / "31_degree_aware_and_multiplex_baselines.py"), "--package-root", str(root), "--output", str(directories["degree"]), "--bootstrap", str(args.bootstrap)])
    run([python, str(code / "32_cross_release_boundary_sensitivity.py"), "--package-root", str(root), "--output", str(directories["boundary"])])
    score_path = directories["degree"] / "degree_aware_full_scores.csv.gz"
    run([python, str(code / "33_pu_label_sensitivity.py"), "--scores", str(score_path), "--output", str(directories["pu"]), "--replicates", str(args.pu_replicates)])
    run([python, str(code / "34_clustered_top100_inference.py"), "--package-root", str(root), "--scores", str(score_path), "--output", str(directories["cluster"]), "--replicates", str(args.cluster_replicates), "--null-permutations", str(args.null_permutations)])
    run([python, str(code / "35_summarize_q2_upgrade.py"), "--upgrade-root", str(root)])
    run([python, str(code / "validate_q2_upgrade.py"), "--upgrade-root", str(root), "--snapshot-root", str(root.parent / "00_original_snapshot_2026-10-13"), "--original-root", str(root.parent.parent)])
    # Refresh the file manifest so it includes the validation report.
    run([python, str(code / "35_summarize_q2_upgrade.py"), "--upgrade-root", str(root)])


if __name__ == "__main__":
    main()
