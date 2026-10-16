from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upgrade-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path)
    args = parser.parse_args()
    root = args.upgrade_root
    results = root / "results_upgrade"
    checks: dict[str, bool | int | str] = {}

    metrics = pd.read_csv(results / "01_degree_aware_multiplex" / "degree_aware_cross_release_metrics.csv")
    checks["six_cancers"] = metrics.cancer.nunique() == 6
    checks["seven_methods"] = metrics.method.nunique() == 7
    checks["one_row_per_cancer_method"] = len(metrics) == 42 and not metrics[["cancer", "method"]].duplicated().any()
    checks["finite_bounded_metrics"] = bool(np.isfinite(metrics[["ROC_AUC", "AUPRC"]]).all().all() and ((metrics[["ROC_AUC", "AUPRC"]] >= 0) & (metrics[["ROC_AUC", "AUPRC"]] <= 1)).all().all())

    boundary = pd.read_csv(results / "02_cross_release_boundary" / "cross_release_boundary_metrics.csv")
    primary = metrics[metrics.method.isin(["RWR-STRING", "DCPI-RWR performance"])][["cancer", "method", "AUPRC"]]
    diagnostic = boundary[(boundary.analysis == "seed-boundary diagnostic") & (boundary.scenario == "historical seeds")][["cancer", "method", "AUPRC"]]
    aligned = primary.merge(diagnostic, on=["cancer", "method"], suffixes=("_primary", "_boundary"))
    checks["boundary_original_matches_primary"] = bool(np.allclose(aligned.AUPRC_primary, aligned.AUPRC_boundary, rtol=0, atol=1e-12))

    pu = pd.read_csv(results / "03_pu_sensitivity" / "pu_ascertainment_replicates.csv.gz")
    incomplete = pu[pu.retained_positive_fraction < 1.0]
    counts = incomplete.groupby(["cancer", "mechanism", "retained_positive_fraction", "method"]).size()
    checks["PU_1000_replicates_per_incomplete_scenario"] = bool((counts == 1000).all())

    cluster_reps = pd.read_csv(results / "04_clustered_inference" / "top100_clustered_bootstrap_replicates.csv.gz")
    counts = cluster_reps.groupby(["method", "cluster"]).size()
    checks["cluster_bootstrap_10000_per_method_unit"] = bool((counts == 10000).all())
    null = pd.read_csv(results / "04_clustered_inference" / "LUAD_degree_matched_seed_null_high_resolution.csv.gz")
    checks["LUAD_null_2000"] = len(null) == 2000

    figures = list(results.rglob("*.pdf")) + list(results.rglob("*.png"))
    checks["eight_nonempty_figure_files"] = len(figures) == 8 and all(path.stat().st_size > 1000 for path in figures)

    snapshot_manifest = pd.read_csv(args.snapshot_root / "ORIGINAL_SNAPSHOT_MANIFEST.sha256.tsv", sep="\t")
    mismatch = 0
    for row in snapshot_manifest.itertuples(index=False):
        path = args.snapshot_root / str(row.relative_path)
        if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path).lower() != str(row.sha256).lower():
            mismatch += 1
    checks["snapshot_manifest_rows"] = len(snapshot_manifest)
    checks["snapshot_hash_mismatches"] = mismatch
    checks["snapshot_integrity"] = mismatch == 0
    if args.original_root is not None:
        original_mismatch = 0
        for row in snapshot_manifest.itertuples(index=False):
            path = args.original_root / str(row.relative_path)
            if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path).lower() != str(row.sha256).lower():
                original_mismatch += 1
        checks["original_vs_snapshot_hash_mismatches"] = original_mismatch
        checks["original_still_matches_snapshot"] = original_mismatch == 0

    passed = all(value is True or isinstance(value, (int, str)) for value in checks.values()) and all(
        value for value in checks.values() if isinstance(value, bool)
    )
    payload = {"status": "PASS" if passed else "FAIL", "checks": checks}
    (root / "UPGRADE_VALIDATION.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
