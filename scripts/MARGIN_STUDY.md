# NCFM angular margin study

The two frozen-teacher diagnostic outputs are `coverage_results.json` and
`coverage_teacher_fixedstats.json`. The former teacher belongs to the
26479 teacher bank and uses mean [0.741, 0.533, 0.706], std
[0.402, 0.821, 0.407]. The latter belongs to the 2210 bank and uses
mean [0.7405449443, 0.5329821482, 0.7058288200], std
[0.1214735017, 0.1741895871, 0.1224783296].
Cross-bank comparisons are held-out by checkpoint hash, although identical
initialization/training ancestry has not been ruled out.

The DD hashes identify the separately trained artifacts. For the 26479 DD,
the cross-bank teacher's real/DD 10th-percentile angular margins averaged
across classes were 0.0537/0.0731. For the 2210 DD, the other teacher's
real/DD means were 0.0437/0.0692. These are training-split diagnostics,
not downstream test accuracy.

As of 2026-11-26, the 20 trained-checkpoint SHA256 sets have no overlap.
The historical 26479 teacher uses substantially different channel standard
deviations than the independently calculated statistics used in 2210. This
study retains its historical statistics to stay consistent with that bank;
the cross-bank evaluation uses each teacher's own preprocessing.

The optional auxiliary is selected with NCFM_MARGIN_MODE=off, overall, or
rival. The fixed targets are computed from train split only and identify each
teacher and data file by SHA256. The default off path keeps NCFM CF loss.
The rival method additionally matches class-versus-rival angular quantiles;
the overall control matches class-versus-strongest-rival quantiles. The
coefficient is a fixed exploratory value (100), not a tuned or proven choice.
Evaluate checkpoints using validation only; report the test confusion matrix,
accuracy, class recall and balanced accuracy at the validation-selected epoch.
Existing normalization choices are retained to reproduce the 26479 bank,
not claimed to be the actual per-channel standard deviations.

Only GPU jobs and results actually recorded separately count as experiments.

## Recorded execution

- Targets: job 203745, completed using one RTX 4090. Each teacher used the
  same 512 training examples per class, seed 7341. The v2 targets select
  rivals using logits and compute angular-margin quantiles.
- Smoke: job 203721, all three arms completed with two distillation iterations
  and two student epochs. Job 203747 additionally verified the v2 targets.
  Smoke accuracies do not measure method performance.
- The first parallel launch (203749) failed: the old entry point overwrote
  Slurm's assigned CUDA_VISIBLE_DEVICES and all tasks selected device 0.
  The entry points now preserve Slurm's allocation. Jobs 203762, 203781 and
  203785 exposed Slurm-visible 4090 devices with low actual free memory;
  the guard rejected those devices. One job on 203762 stalled and was canceled.
- H100 smoke job 203789 gates the formal array 203793. The array has 3 arms,
  3 seeds per arm, 20,000 distillation steps, and 5 validation-selected
  2,000-epoch evaluations per run. Job 203810 collects completed metrics and
  attempts a GitHub push after the array finishes. These are queued, not results.
