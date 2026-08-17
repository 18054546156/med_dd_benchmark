# HoP-TM Table 1/2 Reproduction

This track reproduces the original HoP-TM paper before any NCFM or medical
benchmark extension.

## Source

- Repository: https://github.com/Bian-jh/HoP-TM
- Paper: High-Order Progressive Trajectory Matching for Medical Image Dataset Distillation
- Local checkout used for source inspection: `C:/Users/Administrator/Documents/project/hop_tm_original`

## Official matrix

| Paper table | Dataset | Model | Input size | IPC values |
|---|---|---|---|---|
| Table 1 | PathMNIST | ConvNet | 32x32 | 1, 5, 10, 100, 1000 |
| Table 2 | COVID19-CXR | ConvNetD5 | 112x112 | 1, 5, 10, 50 |

For each dataset, generate 100 teacher trajectories, run 10000 distillation
iterations, and evaluate five random student initializations. Use the official
YAML files under `exp_configs/` from the source checkout without replacing
them with the adapted benchmark YAML files.

## Expected paper reference values

PathMNIST (Table 1, Ours):

```text
IPC 1:    47.71 +/- 1.22
IPC 5:    73.92 +/- 0.66
IPC 10:   77.23 +/- 0.65
IPC 100:  84.82 +/- 0.40
IPC 1000: 89.86 +/- 0.36
```

COVID19-CXR (Table 2, Ours):

```text
IPC 1:   69.71 +/- 0.87
IPC 5:   80.81 +/- 0.38
IPC 10:  84.42 +/- 0.26
IPC 50:  87.71 +/- 0.24
```

These are comparison targets, not locally entered results. A result is valid
only when its buffer, distillation log, saved images/labels/learning rate, and
five-run evaluation log are present.

## Separation rule

Do not place this experiment under `configs/hop_tm/{pathmnist,covid,kvasir}`
or under existing medical-adaptation result directories. On HPC use a separate
root such as:

```text
experiments/hop_table12_original/
  source/
  buffers/
  distill/
  evaluation/
  logs/
  manifests/
```

