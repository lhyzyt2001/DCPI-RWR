from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(script: Path, *arguments: str):
    command = [sys.executable, str(script), *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    code = root / "code_upgrade2"
    results = root / "results_upgrade2"
    results.mkdir(parents=True, exist_ok=True)
    run(code / "36_adaptive_supra_and_seed_weighting.py", "--package-root", str(root), "--output", str(results / "01_adaptive_supra"))
    run(code / "37_context_layer_ablation.py", "--package-root", str(root), "--output", str(results / "02_context_layers"))
    run(code / "38_propensity_pu_stack.py", "--package-root", str(root), "--adaptive-results", str(results / "01_adaptive_supra"), "--output", str(results / "03_propensity_pu"))
    run(code / "39_tcga_expression_anchored_diffusion.py", "--package-root", str(root), "--expression", str(args.expression), "--output", str(results / "04_expression_anchored"))
    run(code / "41_clustered_inference.py", "--package-root", str(root), "--results", str(results), "--output", str(results / "05_clustered_inference"))
    run(code / "40_summarize_fivepoint_upgrade.py", "--results", str(results), "--output", str(results / "00_summary"))


if __name__ == "__main__":
    main()

