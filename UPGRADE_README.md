# Robustness and adaptive-extension analyses

This repository includes the final robustness and adaptive extensions used by the manuscript. They extend the primary DCPI-RWR analysis without replacing the locked Open Targets 26.03-to-26.06 endpoint.

## Included extensions

- deeper non-DCPI neural comparators under leakage-controlled outer folds;
- centrality-bias audits and degree-matched analyses;
- gene-clustered uncertainty for Top-100 recovery;
- leave-one-cancer-out transfer;
- adaptive supra-network RWR and evidence-weighted restart;
- propensity-proxy positive–unlabelled sensitivity;
- network threshold and edge-dropout sensitivity;
- LUAD context-layer and expression-anchored sensitivity analyses.

## Run commands

From the repository root:

```bash
python code_upgrade/run_q2_upgrade.py --upgrade-root .
python code_upgrade2/run_fivepoint_upgrade.py --package-root . --expression path/to/TCGA.LUAD.sampleMap_HiSeqV2/HiSeqV2
```

See `UPGRADE_RESULTS_REPORT.md`, `results_upgrade2/00_summary/FIVE_POINT_UPGRADE_REPORT.md` and `docs/RUN_ORDER.md` for interpretation and ordering. External raw inputs are not redistributed.

