#!/usr/bin/env python3
"""Validate the public DCPI-RWR release and optionally rebuild its manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.sha256.tsv"
VALIDATION = ROOT / "PUBLIC_RELEASE_VALIDATION.json"
TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".cff", ".py", ".tsv", ".csv"}
FORBIDDEN = {
    "previous DOI": "10.5281/zenodo.23307573",
    "previous repository": "DCPI-RWR-gene-prioritization",
    "local user path": r"C:\Users\21215",
    "local analysis path": r"D:\文",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path != MANIFEST
    )


def validate_content() -> dict[str, object]:
    files = release_files()
    python_errors: list[str] = []
    json_errors: list[str] = []
    forbidden_hits: list[str] = []

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8-sig"), rel, "exec")
            except Exception as exc:  # pragma: no cover - release audit
                python_errors.append(f"{rel}: {exc}")
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                json_errors.append(f"{rel}: {exc}")
        if path.resolve() != Path(__file__).resolve() and path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 5_000_000:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for label, pattern in FORBIDDEN.items():
                if pattern.lower() in text.lower():
                    forbidden_hits.append(f"{rel}: {label}")

    cptac = json.loads(
        (ROOT / "results/18_dcpi_rwr_external/CPTAC_manifest.json").read_text(encoding="utf-8")
    )
    with (ROOT / "results/21_constrained_dcpi_rwr/DCPI_RWR_bias_aware_current26_locked_top20.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        expected_genes = [row["Gene"] for row in csv.DictReader(handle)]
    cptac_match = cptac["requested_genes"] == expected_genes
    cptac_description_ok = "DCPI-RWR bias-aware" in cptac["candidate_selection"]

    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    expected_repo = "https://github.com/lhyzyt2001/DCPI-RWR"
    zenodo_repo_ok = any(
        item.get("identifier") == expected_repo for item in zenodo.get("related_identifiers", [])
    )
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_repo_ok = expected_repo in citation_text

    oversized = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if path.stat().st_size >= 100 * 1024 * 1024
    ]
    result = {
        "status": "PASS",
        "repository": "lhyzyt2001/DCPI-RWR",
        "version": "1.0.0",
        "file_count_excluding_manifest": len(files),
        "total_bytes_excluding_manifest": sum(path.stat().st_size for path in files),
        "python_file_count": sum(path.suffix == ".py" for path in files),
        "python_syntax_errors": python_errors,
        "json_errors": json_errors,
        "forbidden_metadata_hits": forbidden_hits,
        "oversized_files": oversized,
        "cptac_top20_exact_order_match": cptac_match,
        "cptac_candidate_description_corrected": cptac_description_ok,
        "zenodo_repository_url_correct": zenodo_repo_ok,
        "citation_repository_url_correct": citation_repo_ok,
    }
    failed = (
        python_errors
        or json_errors
        or forbidden_hits
        or oversized
        or not cptac_match
        or not cptac_description_ok
        or not zenodo_repo_ok
        or not citation_repo_ok
    )
    if failed:
        result["status"] = "FAIL"
    return result


def write_manifest() -> None:
    if VALIDATION.exists():
        VALIDATION.unlink()
    validation = validate_content()
    if validation["status"] != "PASS":
        raise SystemExit(json.dumps(validation, indent=2))

    files_before_validation = release_files()
    validation["file_count_excluding_manifest"] = len(files_before_validation) + 1
    base_bytes = sum(path.stat().st_size for path in files_before_validation)
    previous_size = -1
    while True:
        payload = json.dumps(validation, indent=2) + "\n"
        payload_size = len(payload.encode("utf-8"))
        validation["total_bytes_excluding_manifest"] = base_bytes + payload_size
        if payload_size == previous_size:
            break
        previous_size = payload_size
    VALIDATION.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    rows = []
    for path in release_files():
        rel = path.relative_to(ROOT).as_posix()
        rows.append(f"{rel}\t{path.stat().st_size}\t{sha256(path)}")
    MANIFEST.write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_manifest() -> None:
    if not MANIFEST.exists():
        raise SystemExit("MANIFEST.sha256.tsv is missing")
    errors = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        rel, size_text, expected_hash = line.split("\t")
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing: {rel}")
            continue
        if path.stat().st_size != int(size_text):
            errors.append(f"size mismatch: {rel}")
        if sha256(path) != expected_hash:
            errors.append(f"hash mismatch: {rel}")
    current = {path.relative_to(ROOT).as_posix() for path in release_files()}
    listed = {line.split("\t", 1)[0] for line in MANIFEST.read_text(encoding="utf-8").splitlines()}
    for rel in sorted(current - listed):
        errors.append(f"unlisted: {rel}")
    for rel in sorted(listed - current):
        errors.append(f"listed but absent: {rel}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(listed)} released files.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rebuild validation and manifest")
    args = parser.parse_args()
    if args.write:
        write_manifest()
    verify_manifest()


if __name__ == "__main__":
    main()
