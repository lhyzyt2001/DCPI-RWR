# Five-point DCPI-RWR upgrade: results report

## Preservation boundary

This second-stage upgrade was executed only under `01_DCPI_RWR_Q2_upgrade/code_upgrade2` and `results_upgrade2`. The original snapshot, first-stage upgrade, GitHub repository and Zenodo record were not modified.

## Implemented modules

1. Cancer-adaptive implicit supra-network RWR with historical-only parameter selection.
2. Open Targets clinical-evidence-strength weighted restart and a LUAD TISCH activity-reweighted STRING layer.
3. Cross-fitted diffusion features with a degree-propensity-proxy bagged PU stack.
4. TCGA-LUAD patient-expression-anchored dual-source diffusion.
5. Explicit input and interpretation boundaries for unavailable patient mutation/CNV/methylation and non-LUAD single-cell data.

## Multicancer cross-release summary

| Method | Mean AUPRC | Mean ROC-AUC | Pooled Top-100 hits | Mean hub fraction |
|---|---:|---:|---:|---:|
| RWR-STRING | 0.0526 | 0.9080 | 34 | 0.7900 |
| Adaptive-supra performance | 0.0457 | 0.9006 | 41 | 0.7150 |
| Adaptive-supra evidence-weighted | 0.0443 | 0.9006 | 39 | 0.7050 |
| Adaptive-supra bias-aware | 0.0432 | 0.8981 | 35 | 0.6867 |
| Degree-propensity-weighted bagged PU stack | 0.0197 | 0.6163 | 22 | 0.2150 |

## LUAD context-layer ablation

| method | ROC_AUC | AUPRC | score_degree_spearman | top100_hits | top100_precision | top100_hub_fraction |
|---|---|---|---|---|---|---|
| STRING context: Myeloid_DC | 0.8998 | 0.0349 | 0.9693 | 7 | 0.0700 | 0.6300 |
| STRING context: Mast | 0.9004 | 0.0346 | 0.9695 | 7 | 0.0700 | 0.6300 |
| STRING context: B_Plasma | 0.8994 | 0.0342 | 0.9693 | 6 | 0.0600 | 0.6300 |
| STRING context: Fibrovascular | 0.8986 | 0.0342 | 0.9692 | 7 | 0.0700 | 0.6100 |
| STRING context: Epithelial | 0.8991 | 0.0341 | 0.9693 | 7 | 0.0700 | 0.6100 |
| STRING context: T_cell | 0.8991 | 0.0341 | 0.9693 | 7 | 0.0700 | 0.6300 |
| STRING unweighted | 0.8947 | 0.0331 | 0.9699 | 6 | 0.0600 | 0.5900 |

## LUAD expression-anchored analysis

| method | ROC_AUC | AUPRC | top100_hits | top100_precision | top100_hub_fraction |
|---|---|---|---|---|---|
| Evidence-weighted clinical RWR | 0.8961 | 0.0324 | 6 | 0.0600 | 0.6100 |
| Expression-anchored dual-source consensus | 0.7934 | 0.0238 | 6 | 0.0600 | 0.4600 |
| TCGA expression-aberration diffusion | 0.6794 | 0.0074 | 0 | 0.0000 | 0.0100 |

## Objective interpretation

The highest mean cross-release AUPRC was obtained by **RWR-STRING** (0.0526), compared with 0.0526 for raw STRING-RWR. The largest pooled Top-100 recovery was **Adaptive-supra performance** (41 incident genes). These are descriptive locked-release comparisons; superiority requires clustered uncertainty analysis before a confirmatory claim.

The TCGA module is deliberately labelled expression-anchored dual-source diffusion. It must not be described as patient-specific multi-omics or directional regulatory diffusion because mutation, CNV, methylation and a directed interaction network were not available in the local input package.

## Gene-clustered Top-100 inference

| method | observed_difference | ci_lower | ci_upper | bootstrap_p_two_sided | cluster_sign_flip_p_two_sided |
|---|---|---|---|---|---|
| Adaptive-supra performance | 7.0000 | -5.0000 | 19.0000 | 0.2988 | 0.3397 |
| Adaptive-supra bias-aware | 1.0000 | -10.0000 | 12.0000 | 0.9337 | 1.0000 |
| Adaptive-supra evidence-weighted | 5.0000 | -6.0000 | 17.0000 | 0.4310 | 0.5070 |
| Degree-propensity-weighted bagged PU stack | -12.0000 | -28.0000 | 3.0000 | 0.1398 | 0.1775 |

No upgraded method had a gene-clustered 95% interval excluding zero. Top-100 differences are therefore descriptive and must not be presented as statistically established superiority.
