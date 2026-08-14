# Real Phase 1 Protocol

The formal diagnostics use prepared PathMNIST, COVID, and Kvasir data only.
There is no toy-data fallback.

## Required evidence

- `data/prepared/<dataset>/statistics.json` generated from the train split
- shared `train`, `val`, and `test` split contract
- NCFM trained teacher checkpoint with a recorded SHA-256
- NCFM synthetic tensor for experiments that compare condensed data
- at least 5 independent frequency-bank replicas
- `protocol.json`, `command.txt`, `results.json`, and Slurm stdout/stderr

## Experiments currently implemented

- `E1.1`: independent MC frequency banks comparing fixed real-train NCFM
  teacher features against one explicitly declared real NCFM synthetic artifact,
  including a `T=256,512,1024,2048,4096,8192` sweep
- `E1.2`: paired MC versus scrambled Sobol/QMC banks on the same real/synthetic
  feature pair and seed indices
- `E2.1`: held-out-bank stability of real NCFM synthetic data
- `E2.2`: proposal/importance-weight mechanism control on real teacher features
- `E4.1`: CF measurements recorded for later pairing with a common downstream evaluator

`E2.1` is explicitly a diagnostic proxy because the original NCFM code samples
frequencies inside each loss call and does not persist a training bank. It must
not be reported as proof of frequency overfitting without instrumenting and
rerunning NCFM with an auditable training bank.

## Experiments requiring a new real condensation run

- `E3.1`: at least five real NCFM condense seeds, with identical input/config
  except initialization seed, plus controlled evaluation for each artifact.
- `E3.2`: the released NCFM baseline has `sampling_net=false`; this is
  `not_applicable` to the baseline. A learned-frequency variant must be run as
  a separately named ablation and cannot be silently merged into baseline.
- `E4.2`: pixel-space matching is a different condenser objective. It requires
  a separately instrumented real run with the same teacher, data, optimizer,
  seeds, and downstream evaluator. It cannot be inferred from feature-space
  CF loss.

The legacy toy scripts are never used as evidence for these experiments.
