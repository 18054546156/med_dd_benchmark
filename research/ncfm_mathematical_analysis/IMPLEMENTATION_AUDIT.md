# NCFM Formal Analysis Implementation Audit

Audit date: 2026-08-14

This document distinguishes implemented experimental machinery from measured
HPC evidence. No accuracy, effect, or NCFM defect is considered established
until the referenced real-data artifacts pass every formal gate.

## Current code boundary

- Branch: `codex/formal-real-benchmark-20260814`
- Audited commit: `dd65950`
- Formal HPC worktree:
  `/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark_worktrees/formal-20260814`
- Mathematical output root:
  `/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark/research/ncfm_mathematical_analysis`

## Data gate

The production preflight must:

1. compute PathMNIST, COVID, and Kvasir statistics from the train split only;
2. reject duplicate decoded RGB pixels;
3. enforce PIL bicubic resize before `ToTensor`, with raw train tensors in
   `[0,1]`;
4. write `data_audit/current_ready.json` with resolved prepared paths and
   manifest/statistics SHA-256 values;
5. verify that audit without rewriting it before any producer starts;
6. bind the audit SHA into every NCFM and HoP run manifest.

No post-production stage may regenerate `statistics.json`, because that would
invalidate the hash used by the producer manifests.

## Phase 1 implementation

The formal matrix contains eight logical diagnostics for each of three real
datasets, for 24 logical runs total:

| ID | Measurement | Required interpretation |
|---|---|---|
| E1.1 | Independent MC banks and T sweep | Frequency-estimator variability |
| E1.2 | Paired MC versus scrambled QMC | Estimator comparison only |
| E2.1 | Independent holdout-bank stability | Diagnostic proxy, not proof of training-bank overfit |
| E2.2 | Exact importance correction control | Mechanism control, not a released-baseline defect |
| E3.1 | Five independent baseline condenser seeds | Initialization sensitivity with controlled evaluation |
| E3.2 | Learned-frequency comparison | Not applicable to fixed-frequency baseline; tested as a separate Phase 2 variant |
| E4.1 | Class-wise CF value versus downstream accuracy | Requires registered pairs and correlation evidence |
| E4.2 | Feature-CF versus pixel-mean control | Separate real condenser runs with matched seeds |

The 24 logical diagnostics are grouped into three Slurm array tasks to respect
`MaxSubmitJobsPerUser=15`. All CF diagnostics use class-wise measurements and
class-equal aggregation.

## Phase 2 implementation

Three real condenser variants are implemented:

1. scrambled-QMC frequency sampling;
2. exact importance sampling with untruncated `p/q` weights;
3. learned-frequency minimax sampling with synchronized DDP proposal updates.

Each variant is run for three datasets and five independent condenser seeds.
Every resulting synthetic artifact is evaluated with ConvNet and ResNet18,
giving `3 variants x 3 datasets x 5 seeds x 2 architectures = 90` controlled
evaluation entries. QMC, importance, and holdout-certificate diagnostics are
also run as nine separate logical measurements.

The certificate is conditional on a fixed feature pair and independent
holdout frequencies. It is not represented as a certificate for the complete
optimization process, unseen backbones, or unseen datasets.

## HoP medical stability gate

PathMNIST keeps its configured image learning rate. COVID and Kvasir screen
`lr_img = 100, 10, 1, 0.1` from largest to smallest on the newly generated,
run-scoped buffer. Selection uses only finite completion of a 200-iteration
short run and never reads validation or test accuracy. The selected value,
candidate logs, and hashes are written under `hop_stability/<RUN_ID>/` and are
bound into the HoP run manifest.

This is disclosed as a medical stability selection. If the selected value is
not the configured value, the result must not be described as an unchanged
raw HoP hyperparameter reproduction.

## Artifact and scheduler contracts

- No synthetic tensor is selected by modification time.
- Every producer has a unique `RUN_ID`.
- Run manifests bind source, config, current data audit, statistics, teacher or
  buffer files, synthetic tensors, command, stdout, and stderr by SHA-256.
- Per-seed variant logs are immutable and separate.
- Staged submission peaks are below 15 submitted records per user.
- Five-seed condenser sweeps run sequentially within one named allocation.
- Large evaluation matrices are partitioned into bounded grouped jobs.

## Evidence still required

The implementation is ready, but the formal scientific conclusion remains
`insufficient_evidence` until all of the following exist on HPC:

- refreshed three-dataset `current_ready.json` and passing CPU/GPU contracts;
- six complete baseline producer manifests (NCFM and HoP on three datasets);
- complete 24-entry Phase 1 matrix;
- complete 12-entry NCFM/HoP controlled evaluation matrix;
- complete nine-entry Phase 2 diagnostic matrix;
- complete 90-entry Phase 2 real-variant evaluation matrix;
- generated `formal_report.json`, `formal_report.md`, tuning plan, and figures.

At that point the report's preregistered effect rules, rather than log
inspection or manually selected examples, determine whether any NCFM defect is
supported or confirmed.
