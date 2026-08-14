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

The production hierarchy is:

```text
ncfm_mathematical_analysis/
  data_audit/                 immutable split/statistics audit snapshots
  submissions/                Slurm dependency records and run tags
  runs/ncfm/<dataset>/<run>/  NCFM commands, hashes, and run manifests
  runs/hop_tm/<dataset>/<run>/ HoP commands, hashes, and run manifests
  hop_stability/<run>/        finite-only HoP image-LR selection evidence
  phase2/                     QMC/importance/certificate diagnostics
  phase2_manifests/           explicit real-variant input manifests
  figures/                    plots generated only from complete JSON evidence
  formal_report.{json,md}
  medical_tuning_plan.{json,md}
```

Teacher checkpoints, replay buffers, synthetic tensors, and Slurm logs remain
under `pretrained_models/`, `buffers/`, `results/`, and `logs/`.  This analysis
directory binds those files by explicit path and SHA-256; it does not duplicate
or select them by modification time.

The staged submission graph is constrained to the cluster's
`MaxSubmitJobsPerUser=15`. Logical arrays are grouped and long seed sweeps run
inside named allocations. Statistics are generated once before production and
are never rewritten after a run manifest has bound their hash.

The formal artifact and controlled-evaluation input manifests remain under
`research/ncfm_medical_analysis/` because they describe the benchmark's
production runs rather than the mathematical analysis outputs.
