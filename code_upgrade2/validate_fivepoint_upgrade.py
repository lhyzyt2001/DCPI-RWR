from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root
    results = root / "results_upgrade2"
    adaptive = pd.read_csv(results / "01_adaptive_supra" / "adaptive_supra_cross_release_metrics.csv")
    oof = pd.read_csv(results / "01_adaptive_supra" / "adaptive_supra_oof_features.csv.gz")
    pu = pd.read_csv(results / "03_propensity_pu" / "propensity_pu_cross_release_metrics.csv")
    context = pd.read_csv(results / "02_context_layers" / "LUAD_context_layer_ablation.csv")
    expression = pd.read_csv(results / "04_expression_anchored" / "LUAD_expression_anchored_cross_release_metrics.csv")
    inference = pd.read_csv(results / "05_clustered_inference" / "top100_clustered_inference.csv")
    metric_values = pd.concat([adaptive[["ROC_AUC", "AUPRC"]], pu[["ROC_AUC", "AUPRC"]]], ignore_index=True)
    checks = {
        "adaptive_6_cancers_x_4_methods": len(adaptive) == 24 and adaptive.cancer.nunique() == 6 and adaptive.method.nunique() == 4,
        "adaptive_oof_complete": len(oof) == 6 * 25242 and not oof[["raw_rwr", "adaptive_performance", "adaptive_bias_aware"]].isna().any().any(),
        "PU_six_cancers": len(pu) == 6 and pu.cancer.nunique() == 6,
        "LUAD_six_contexts_plus_baseline": len(context) == 7,
        "expression_three_methods": len(expression) == 3,
        "bounded_metrics": bool(np.isfinite(metric_values.to_numpy(float)).all() and ((metric_values >= 0) & (metric_values <= 1)).all().all()),
        "clustered_inference_four_methods_two_units": len(inference) == 8 and inference.method.nunique() == 4 and inference.cluster.nunique() == 2,
        "required_manifests": all(
            p.exists()
            for p in [
                results / "01_adaptive_supra" / "adaptive_supra_manifest.json",
                results / "02_context_layers" / "context_layer_manifest.json",
                results / "03_propensity_pu" / "propensity_pu_manifest.json",
                results / "04_expression_anchored" / "expression_anchored_manifest.json",
                results / "00_summary" / "FIVE_POINT_UPGRADE_REPORT.md",
            ]
        ),
    }
    manifest = args.snapshot_root / "ORIGINAL_SNAPSHOT_MANIFEST.sha256.tsv"
    mismatches = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = args.snapshot_root / row["relative_path"]
            if not path.exists() or path.stat().st_size != int(row["bytes"]) or digest(path) != row["sha256"]:
                mismatches.append(row["relative_path"])
    checks["original_snapshot_hash_mismatches"] = len(mismatches)
    checks["original_snapshot_unchanged"] = len(mismatches) == 0
    passed = all(value is True or (isinstance(value, int) and value == 0) for value in checks.values())
    payload = {"status": "PASS" if passed else "FAIL", "checks": checks, "snapshot_mismatches": mismatches}
    (root / "FIVE_POINT_UPGRADE_VALIDATION.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for folder in (root / "code_upgrade2", root / "results_upgrade2"):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                rows.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)})
    pd.DataFrame(rows).to_csv(root / "FIVE_POINT_UPGRADE_MANIFEST.sha256.tsv", sep="\t", index=False, encoding="utf-8-sig")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
