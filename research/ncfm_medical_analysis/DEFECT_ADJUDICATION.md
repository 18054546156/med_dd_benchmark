# NCFM Defect Adjudication Protocol

This file defines the decision rules before the real runs. It prevents a
post-hoc accuracy threshold from being promoted to a defect claim.

## Evidence States

Every hypothesis is reported as one of:

```text
confirmed
supported
not_applicable
insufficient_evidence
failed
```

`supported` means the declared measurement was completed. `confirmed` is
reserved for a hypothesis whose preregistered effect rule and minimum evidence
are both satisfied. A missing run is never treated as a negative result.

## Four Evidence Layers

| Layer | Experiment | Hypothesis | Minimum evidence | Pre-registered effect rule |
|---|---|---|---|---|
| stochastic estimator | E1.1 | CF estimate depends materially on frequency budget | 5 replicas and all declared T values | max/min sweep mean ratio >= 1.20 |
| stochastic estimator | E1.2 | QMC reduces estimator variance relative to MC | 5 paired replicas at T=4096 | mean std-reduction >= 0.20 and paired values are available |
| objective stability | E2.1 | a held-out frequency bank disagrees with the measured bank | 5 replicas | held-out minus measured mean > 0 with a positive 95% bootstrap interval |
| objective correction | E2.2 | proposal correction reduces estimator error | 5 paired replicas | corrected mean absolute error < uncorrected by at least 20% |
| optimization | E3.1 | condensation depends materially on initialization | 5 independent real condensation runs and controlled evals | test-accuracy standard deviation >= 2 percentage points |
| released baseline scope | E3.2 | learned-frequency baseline behavior | released-code audit | `not_applicable` while `sampling_net=false` |
| surrogate validity | E4.1 | CF loss is not predictive of downstream quality | 5 paired synthetic/eval artifacts | absolute Pearson and Spearman association < 0.30 |
| objective comparison | E4.2 | feature-space and pixel-space objectives differ | 5 paired real runs | pairwise mean test-accuracy delta >= 2 percentage points |

These rules identify a measured limitation of the tested configuration. They
do not automatically generalize to every dataset, architecture, or paper
implementation. Dataset-level statements require the same result on the
declared dataset; cross-dataset claims require replication on all three.

## What Counts As A Formal NCFM Defect

A report may use `confirmed` only when:

1. the real dataset, train-only statistics, checkpoint, synthetic artifact and
   controlled evaluator are recorded by SHA-256;
2. the required replicate count and paired manifest entries exist;
3. the exact effect rule above is satisfied;
4. the run did not use toy data, legacy logs, mtime-based artifact selection,
   or an unrecorded configuration change.

Otherwise the report must use `supported` or `insufficient_evidence`.
