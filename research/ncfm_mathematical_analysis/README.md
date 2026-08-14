# NCFM Mathematical Analysis

This directory is the single output root for the auditable mathematical and
mechanism analysis of NCFM. It is separate from
`research/ncfm_medical_analysis`, which stores source protocols, dataset
statistics, and teacher task analysis.

On HPC the default path is:

```text
/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark/research/ncfm_mathematical_analysis
```

Set `NCFM_MATH_ROOT` to override it. Generated artifacts include:

- `runs/`: real Phase 1 diagnostic run directories;
- `phase1_replication_manifest.json`: explicit Phase 1 replication inputs;
- `phase2/`: real Phase 2 mechanism diagnostic outputs;
- `phase2_variant_manifest.json` and `phase2_variant_report.*`: explicit variant evaluations;
- `formal_report.*`, `medical_tuning_plan.*`, and `figures/`: downstream report artifacts.

The formal artifact and controlled-evaluation input manifests remain under
`research/ncfm_medical_analysis/` because they describe the benchmark's
production runs rather than the mathematical analysis outputs.
