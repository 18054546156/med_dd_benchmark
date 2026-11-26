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

The optional auxiliary is selected with NCFM_MARGIN_MODE=off, overall, or
rival. The fixed targets are computed from train split only and identify each
teacher and data file by SHA256. The default off path keeps NCFM CF loss.
The rival method additionally matches class-versus-rival angular quantiles;
the overall control matches class-versus-strongest-rival quantiles. The
coefficient is a proposed value and must be tuned on validation, never test.
Existing normalization choices are retained to reproduce the 26479 bank,
not claimed to be the actual per-channel standard deviations.

Only GPU jobs and results actually recorded separately count as experiments.
