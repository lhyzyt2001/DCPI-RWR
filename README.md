# DCPI-RWR: cross-release multicancer gene prioritization

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release: v1.0.0](https://img.shields.io/badge/release-v1.0.0-5c6bc0.svg)](https://github.com/lhyzyt2001/DCPI-RWR/releases/tag/v1.0.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23408226.svg)](https://doi.org/10.5281/zenodo.23408226)

This repository contains the source code and publication-level derived results for **DCPI-RWR**, a degree-conditioned, pathway-integrated network-diffusion framework, together with a cancer-adaptive supra-network extension and a cross-release multicancer benchmark.

The study asks whether network-based gene prioritization can improve fixed-budget recovery while reducing dependence on highly connected STRING genes. It is not presented as a universally superior predictor, a causal-gene discovery system or a therapeutic-target validation platform.

## Study design

- Open Targets release 26.03 provides historical cancer-specific seed labels and all configuration selection.
- Genes newly receiving clinical-stage evidence in release 26.06 form the locked cross-release endpoint.
- Six cancer scopes are evaluated: LUAD, breast, colorectal, prostate, melanoma and ovarian cancer.
- STRING-RWR, degree baselines, boosted models, RWR-feature graph neural networks, DCPI-RWR and adaptive supra-RWR variants are evaluated under explicit leakage controls.
- AUPRC, ROC-AUC, Top-100 incident-gene recovery, STRING hub fraction, uncertainty and network perturbation are reported jointly.
- A locked bias-aware LUAD Top-20 list is characterized in TCGA-LUAD, GSE68465 and CPTAC-LUAD without using those outcomes for tuning or reranking.

The graph topology was not independently frozen at release 26.03. The design is therefore described as **cross-release evaluation**, not prospective clinical validation.

## Main findings

- In the contemporary LUAD benchmark, STRING-RWR achieved the highest mean AUPRC (0.554), followed by LightGBM + RWR/features (0.536) and XGBoost + RWR/features (0.534). The evaluated RWR-feature graph models ranged from 0.330 to 0.362.
- Across six locked cross-release tasks, STRING-RWR retained the highest mean AUPRC (0.0526).
- Performance-oriented DCPI-RWR recovered 44 pooled Top-100 incident genes versus 34 for STRING-RWR and reduced the mean Top-100 hub fraction from 0.790 to 0.485. The gene-clustered confidence interval for the +10-hit difference included zero.
- Cancer-adaptive supra-RWR recovered 41 pooled incident genes with a mean hub fraction of 0.715, but its mean AUPRC (0.0457) remained below STRING-RWR and its hit-gain interval also included zero.
- Moderate network perturbation preserved cancer-specific ranks (minimum Spearman correlation 0.843).
- LUAD context reweighting produced only small AUPRC changes, and expression-anchored diffusion did not outperform clinical-evidence diffusion.
- VEGFB, PDGFD and EPHA4 showed FDR-significant, directionally concordant RNA differences in TCGA-LUAD and GSE68465. No adjusted survival association remained significant after multiplicity correction.

These results support a performance–centrality trade-off and transparent hypothesis prioritization; they do not establish universal predictive superiority, causality or therapeutic efficacy.

## Repository layout

```text
.
├── code/                 # primary DCPI-RWR and benchmark analyses
├── code_upgrade/         # additional centrality, transfer and uncertainty analyses
├── code_upgrade2/        # adaptive supra-RWR, perturbation and context extensions
├── docs/                 # methods, run order, provenance and interpretation boundaries
├── inputs/               # permitted derived graph objects and benchmark predictions
├── results/              # primary frozen derived results
├── results_upgrade/      # first-stage robustness extensions
├── results_upgrade2/     # final adaptive and sensitivity extensions
├── CITATION.cff
├── LICENSE
├── MANIFEST.sha256.tsv
└── requirements.txt
```

Earlier PU-learning and multiview-GCN outputs are retained only as clearly labelled provenance and sensitivity material. They are not components of the primary DCPI-RWR method.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/lhyzyt2001/DCPI-RWR.git
cd DCPI-RWR
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and PyTorch Geometric wheels must match the operating system and accelerator. The reported analyses used Python 3.11.3 and PyTorch 2.6.0 (CPU build); a GPU is optional.

## Reproduction

The primary execution order and exact command-line arguments are documented in [`docs/RUN_ORDER.md`](docs/RUN_ORDER.md). Principal entry points include:

```bash
python code/25_degree_conditioned_pathway_rwr.py --help
python code/27_loco_transfer_dcpi_rwr.py --help
python code/28_pathway_component_permutation.py --help
python code/26_summarize_biasaware_framework.py --help
python code_upgrade2/run_fivepoint_upgrade.py --help
```

Frozen derived outputs permit claim-level verification without rerunning the most computationally intensive analyses. Access-controlled or third-party raw data are not redistributed; see [`docs/ANALYSIS_SCOPE.md`](docs/ANALYSIS_SCOPE.md) and the provenance records in `docs/` and `results/`.

## Key result locations

- `results/21_constrained_dcpi_rwr`: locked DCPI-RWR configurations and cross-release results.
- `results/18_dcpi_rwr_external`: TCGA, GSE68465 and CPTAC characterization of the fixed bias-aware LUAD candidates.
- `results/20_loco_transfer`: leave-one-cancer-out transfer results.
- `results/22_final_innovation_summary`: manuscript-facing summary tables.
- `results/23_pathway_component_permutation`: degree-conditional pathway-component permutation.
- `results_upgrade2/01_adaptive_supra`: cancer-adaptive supra-RWR configurations and scores.
- `results_upgrade2/03_propensity_pu`: propensity-proxy positive–unlabelled sensitivity analysis.
- `results_upgrade2/04_network_sensitivity`: network threshold and edge-dropout sensitivity.
- `results_upgrade2/05_context_extension`: LUAD context and expression-anchored sensitivity analyses.
- `results/manuscript_figures` and `results_upgrade2/06_figures`: publication figures.

## Integrity and interpretation

`MANIFEST.sha256.tsv` records the byte size and SHA-256 checksum of every released file except the manifest itself and Git metadata. The endpoint is later Open Targets clinical-stage evidence, not driver-gene status. External molecular and survival analyses are secondary characterization and were not used to tune, filter or reorder the fixed candidate list.

Verify the frozen release with:

```bash
python verify_release.py
```

## Citation

Please cite the software release described in [`CITATION.cff`](CITATION.cff). The immutable `v1.0.0` archive is available at [https://doi.org/10.5281/zenodo.23408226](https://doi.org/10.5281/zenodo.23408226).

## License

Source code is released under the [MIT License](LICENSE). Derived tables and figures remain subject to the terms of their underlying public data sources; restricted and third-party raw datasets are not redistributed.

## Contact

Corresponding author: Jihua Feng (`fengjihua@ymu.edu.cn`).

