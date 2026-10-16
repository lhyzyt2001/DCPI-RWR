# DCPI-RWR cross-release benchmark v1.0.0

This is the first independent public release of the complete DCPI-RWR cross-release multicancer benchmark. It has a new repository and an independent Git history; it is not derived from or layered onto the previous DCPI-RWR repository.

## Included

- primary degree-conditioned, pathway-integrated RWR analyses;
- contemporary LUAD comparisons with STRING-RWR, degree, boosted models and RWR-feature graph models;
- locked Open Targets 26.03-to-26.06 evaluation across six cancers;
- cancer-adaptive supra-RWR and evidence-weighted restart analyses;
- gene-clustered uncertainty, leave-one-cancer-out transfer and centrality audits;
- network-threshold, edge-dropout, context-layer and positive–unlabelled sensitivity analyses;
- fixed bias-aware LUAD Top-20 characterization in TCGA-LUAD, GSE68465 and CPTAC-LUAD;
- frozen publication-level tables, figures, provenance metadata and SHA-256 checksums.

## Interpretation boundary

STRING-RWR remains the strongest method by mean cross-release AUPRC. DCPI-RWR improves some fixed-budget recovery and hub-burden trade-offs, but the corresponding gain intervals include zero. The release supports transparent computational prioritization and bias-aware benchmarking, not causal or therapeutic validation.

