# DCPI-RWR independent upgrade: results report

## Scope and preservation boundary

This directory is an independent copy created on 2026-10-13. The original experiment package, manuscript, GitHub repository and Zenodo record were not edited. All new code is under `code_upgrade/`; all new outputs are under `results_upgrade/`.

The upgrade addresses four prespecified priorities: (1) degree-aware and multiplex diffusion comparators; (2) cross-release and network-input boundary sensitivity; (3) positive-unlabelled label-ascertainment sensitivity; and (4) gene- and cancer-clustered Top-100 inference with a higher-resolution LUAD degree-matched seed null.

## 1. Degree-aware and multiplex comparators

The DADA-CD comparator is a documented candidate-degree reference-model reimplementation inspired by Erten et al. (PLoS ONE 2011; doi:10.1371/journal.pone.0021171), not a claim of bit-for-bit reproduction of the historical MATLAB implementation. Multiplex RWR is defined as RWR on a convex mixture of column-normalized layer transition matrices. Hyperparameters were selected using Open Targets 26.03 labels only and evaluated on genes newly labelled by 26.06.

| Method | Mean AUPRC | Mean ROC-AUC | Mean absolute Spearman rho | Pooled Top-100 incident hits | Mean Top-100 hub fraction |
|---|---:|---:|---:|---:|---:|
| RWR-STRING | 0.0526 | 0.9080 | 0.970 | 34 | 0.790 |
| Multiplex-RWR STRING+pathway | 0.0442 | 0.8993 | 0.947 | 40 | 0.720 |
| DCPI-RWR performance | 0.0432 | 0.8937 | 0.944 | 44 | 0.485 |
| Degree-penalized RWR | 0.0414 | 0.9035 | 0.961 | 27 | 0.582 |
| Multiplex-RWR three-view | 0.0356 | 0.8696 | 0.680 | 31 | 0.588 |
| DADA optimistic uniform | 0.0338 | 0.8821 | 0.853 | 23 | 0.322 |
| DADA-CD reference model | 0.0150 | 0.7902 | 0.797 | 10 | 0.010 |

Cancer-specific AUPRC winners were:

- BRCA: RWR-STRING, AUPRC 0.0444.
- CRC: RWR-STRING, AUPRC 0.0902.
- LUAD: Multiplex-RWR three-view, AUPRC 0.0433.
- MEL: RWR-STRING, AUPRC 0.0466.
- OV: Multiplex-RWR STRING+pathway, AUPRC 0.0572.
- PRAD: RWR-STRING, AUPRC 0.0482.

Interpretation: no single upgraded method dominates all cancers. Raw STRING-RWR remains the strongest mean-AUPRC method and wins BRCA, CRC, MEL and PRAD. Multiplex STRING+pathway wins OV, whereas three-view multiplex RWR wins LUAD and has much lower score-degree correlation than raw RWR. This is useful evidence for cancer-specific multiview value, but it does not justify a universal performance-superiority claim.

## 2. Cross-release and network-input boundary

Across cancers, historical-label retention from 26.03 to 26.06 ranged from 0.921 to 1.000. Removed historical positives were absent in LUAD and ranged up to 27 in the other cancers. Replacing historical seeds with only labels that persist in 26.06 changed results little in most cancers, but this is explicitly diagnostic because it uses future information.

| Network perturbation | Method | Mean AUPRC across cancers |
|---|---|---:|
| STRING deterministic edge dropout 10% | DCPI-RWR performance | 0.0432 |
| STRING deterministic edge dropout 10% | RWR-STRING | 0.0520 |
| STRING deterministic edge dropout 20% | DCPI-RWR performance | 0.0430 |
| STRING deterministic edge dropout 20% | RWR-STRING | 0.0509 |
| STRING score >= 0.80 | DCPI-RWR performance | 0.0412 |
| STRING score >= 0.80 | RWR-STRING | 0.0431 |
| STRING score >= 0.90 | DCPI-RWR performance | 0.0394 |
| STRING score >= 0.90 | RWR-STRING | 0.0415 |

The minimum cancer-specific rank correlation with the original packaged graph was 0.843. These analyses support robustness to modest deterministic edge dropout, while stricter STRING thresholds can alter performance. Because a historical STRING snapshot was unavailable, this does not establish fully prospective network freezing.

## 3. Positive-unlabelled label-ascertainment stress test

Predictions were held fixed while 25%, 50% or 75% of incident positives were retained either uniformly (SCAR) or with degree-biased ascertainment. This is a stress test of incomplete outcome ascertainment; it is not a latent-positive estimator.

| Method vs RWR-STRING | Scenarios with positive median delta | Median delta across incomplete-label scenarios |
|---|---:|---:|
| Multiplex-RWR STRING+pathway | 12/36 | -0.0031 |
| Degree-penalized RWR | 2/36 | -0.0031 |
| DCPI-RWR performance | 6/36 | -0.0036 |
| DADA optimistic uniform | 0/36 | -0.0101 |
| Multiplex-RWR three-view | 7/36 | -0.0118 |
| DADA-CD reference model | 0/36 | -0.0251 |

The method ordering is therefore not uniformly robust to incompletely observed positives. LUAD tends to favour the multiview methods, whereas several other cancers continue to favour raw RWR, especially under degree-biased ascertainment.

## 4. Clustered Top-100 inference

The estimand is the pooled difference in newly labelled genes recovered by six cancer-specific Top-100 lists. The primary bootstrap samples unique genes and keeps all cancer-specific contributions for a sampled gene together; the secondary bootstrap samples cancers.

| Method vs RWR-STRING | Observed hit difference | Gene-clustered 95% CI | Bootstrap P | Cluster sign-flip P |
|---|---:|---:|---:|---:|
| DADA optimistic uniform | -11 | -21 to -1 | 0.033 | 0.039 |
| DADA-CD reference model | -24 | -39 to -10 | 0.000 | 0.001 |
| Degree-penalized RWR | -7 | -15 to 0 | 0.051 | 0.093 |
| Multiplex-RWR STRING+pathway | 6 | -7 to 19 | 0.389 | 0.450 |
| Multiplex-RWR three-view | -3 | -15 to 8 | 0.669 | 0.742 |
| DCPI-RWR performance | 10 | -6 to 26 | 0.256 | 0.290 |

DCPI-RWR recovered 10 more pooled Top-100 incident hits than raw RWR, but the gene-clustered interval included zero. Thus the gain is descriptive rather than statistically secure. DADA-CD and the optimistic merge reduced centrality dependence but lost incident-hit recovery.

The 2,000-replicate LUAD degree-matched seed null gave observed AUPRC 0.0278, null mean 0.0163, 95% null interval 0.0097–0.0407, and empirical upper-tail P=0.065. This is suggestive but does not pass a conventional 0.05 threshold.

## Overall judgment

The upgrade improves methodological depth and reviewer-facing transparency. Its strongest defensible contribution is not universal superiority over RWR, but a controlled analysis of when multiview diffusion can trade predictive discrimination against network-centrality dependence under cross-release evaluation. The new evidence is internally coherent: some cancers benefit from pathway or three-view integration, while raw RWR remains difficult to beat and the uncertainty of Top-100 gains is now quantified correctly.

For a revised manuscript, retain the negative and heterogeneous results. Do not claim that DCPI-RWR is globally superior, degree independent, prospectively validated, or a causal/therapeutic target predictor. A stronger framing is a degree-aware multiview diffusion benchmarking framework with cancer-specific utility and explicit ascertainment/centrality limits.
