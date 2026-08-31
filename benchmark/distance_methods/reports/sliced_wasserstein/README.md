# Sliced-Wasserstein Pre-experiment Audit

This cleanroom audits the existing `fixed-U4096 + sliced Wasserstein` signal
before any new condensation run is allowed.

The first stage is deliberately offline:

1. compare the historical interpolation estimator with exact one-dimensional OT;
2. measure NCFD/SW gradient dose and conflict on the same synthetic artifact;
3. test projection-count and projection-seed stability;
4. compare real-synthetic SW with a matched-size real-real null;
5. run equal-dose learner-utility tests only if the preceding gates pass.

No script scans for the latest artifact. Input paths and SHA256 values are
locked in `configs/path_seed0.yaml`.

## Outputs

The offline runner writes:

- `p0_estimator_validation.csv`
- `p1_gradient_interaction.csv`
- `p2_projection_stability.csv`
- `p3_matched_null.csv`
- `gate_summary.json`
- `run_manifest.json`

These results are diagnostics, not a new DD result.

The completed audit also contains:

- `results/path_seed0_projection_terminal/`: the terminal K=8192 stability gate;
- `results/path_seed0_equal_dose_utility/`: equal-pixel-dose CF/SW controls;
- `results/path_seed0_learner_proxy/`: fixed-final-epoch fresh-learner checks;
- `SW_PREEXPERIMENT_ANALYSIS.executed.ipynb`: executed analysis and figures;
- `SW_PREEXPERIMENT_DECISION.md`: the final go/no-go decision.

Final decision: the SW discrepancy is real, but its improvement does not
consistently transfer to fresh learners. No new CF+SW condensation is approved.
