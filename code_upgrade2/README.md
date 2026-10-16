# DCPI-RWR second-stage five-point upgrade

This directory is an independent extension of the first-stage Q2 upgrade. It does not modify the original experiment snapshot, GitHub repository or Zenodo record.

## Method nomenclature

- **DCPI-RWR** means *degree-conditioned pathway-integrated random walk with restart*. It denotes the rank-fusion method that combines raw STRING-RWR with degree-bin-conditioned network-view percentiles. “Pathway-integrated” refers to score-level integration of the KEGG-derived view, not to a claim that pathway information improves every cancer.
- **STRING-RWR** means conventional RWR on the STRING functional-association graph. Some frozen machine-readable tables retain the legacy identifier `RWR-STRING`; the two labels denote the same reference method.
- **Cancer-adaptive supra-RWR** denotes the implicit supra-network RWR whose restart, inter-layer switching probability, layer weights and restart-seed mode are selected separately for each cancer using Open Targets 26.03 labels only. Earlier output tables abbreviate this family as `Adaptive-supra`.
- **Evidence-weighted restart** is an optional restart-seed weighting mode within cancer-adaptive supra-RWR, not a mandatory component of every selected configuration.
- **Cell-type layer** denotes the packaged TISCH-derived cell-type similarity view. For LUAD context analyses, STRING is additionally reweighted by TISCH cell-group activity; this is not a de novo regulatory network.

These are network-diffusion methods or prioritization strategies rather than deep-learning models.

## Modules

1. `36_adaptive_supra_and_seed_weighting.py`: cancer-adaptive implicit supra-network RWR and Open Targets evidence-strength restart weighting.
2. `37_context_layer_ablation.py`: LUAD TISCH cell-group activity reweighting of STRING.
3. `38_propensity_pu_stack.py`: cross-fitted diffusion features with a degree-propensity-proxy bagged PU sensitivity model.
4. `39_tcga_expression_anchored_diffusion.py`: TCGA-LUAD patient-expression-anchored dual-source diffusion.
5. `41_clustered_inference.py`: gene- and cancer-clustered Top-100 inference against STRING-RWR.
6. `40_summarize_fivepoint_upgrade.py` and `validate_fivepoint_upgrade.py`: report and integrity validation.

## Run

```powershell
python code_upgrade2/run_fivepoint_upgrade.py `
  --package-root "." `
  --expression "path/to/TCGA.LUAD.sampleMap_HiSeqV2/HiSeqV2"
```

## Interpretation boundaries

- All model selection uses Open Targets 26.03 labels; incident 26.06 labels are reserved for locked cross-release evaluation.
- The LUAD context layer is STRING reweighted by packaged TISCH expression activity, not a de novo regulatory network.
- The PU model uses degree as a label-observation propensity proxy. It is a sensitivity analysis, not an identified SAR-PU estimator.
- The TCGA module uses patient expression only. Mutation, CNV and methylation were not available locally, and the packaged STRING graph is effectively undirected. Therefore the module is an exploratory dual-source consensus, not a full NetICS-style directional multi-omics analysis.
