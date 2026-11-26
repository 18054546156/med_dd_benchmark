"""Summarize completed Slurm tasks and paired downstream metrics."""
import argparse
import json
from math import isclose, isfinite
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess


MODES = ("off", "overall", "rival")
SEEDS = (0, 1, 2)
REPEATS = 5


def read_metrics(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != REPEATS:
        raise ValueError("expected five student records")
    for row in records:
        matrix = row["confusion"]
        if len(matrix) != 9 or any(len(r) != 9 for r in matrix):
            raise ValueError("expected a 9-class confusion matrix")
        totals = [sum(r) for r in matrix]
        if any(n <= 0 for n in totals) or sum(totals) != 7180:
            raise ValueError("unexpected PathMNIST test size or class support")
        recall = [matrix[c][c] / n for c, n in enumerate(totals)]
        accuracy = 100 * sum(matrix[c][c] for c in range(9)) / sum(totals)
        for key, expected in (("test_accuracy", accuracy), ("test_bacc", 100 * mean(recall))):
            if not isfinite(row[key]) or not isclose(row[key], expected, abs_tol=1e-4):
                raise ValueError(f"{key} does not match confusion matrix")
        if row["selected_epoch"] not in range(100, 2001, 100):
            raise ValueError("invalid validation-selected epoch")
    return records


def collect(args):
    accounting = subprocess.run(
        ["sacct", "-X", "-j", args.job_id, "-n", "-P",
         "--format=JobID,State,ExitCode,Elapsed,NodeList"],
        text=True, capture_output=True, check=False,
    )
    lines = accounting.stdout.splitlines()
    states = {}
    for line in lines:
        job, state, code, *_ = line.split("|")
        states[job] = (state, code)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"job_id": args.job_id, "run_tag": args.run_tag, "training_steps": 20000,
               "eval_epochs": 2000, "expected_seeds": SEEDS, "expected_repeats": REPEATS,
               "sacct": lines, "methods": {}, "paired": {}}
    if accounting.returncode:
        summary["accounting_error"] = accounting.stderr.strip()
    finished = {}
    for index, method in enumerate(MODES):
        runs = []
        finished[method] = {}
        for seed in SEEDS:
            task = f"{args.job_id}_{3 * index + seed}"
            state, code = states.get(task, ("UNKNOWN", ""))
            entry = {"seed": seed, "task_id": task, "slurm_state": state, "status": "unverified"}
            runs.append(entry)
            if (state, code) != ("COMPLETED", "0:0"):
                continue
            path = args.study_root / "outputs" / args.run_tag / method / f"seed{seed}" / "evaluation.json"
            try:
                metrics = read_metrics(path)
            except (OSError, TypeError, ValueError, KeyError) as error:
                entry.update(status="invalid_metrics", error=str(error))
                continue
            destination = args.out / f"{method}_seed{seed}.json"
            shutil.copyfile(path, destination)
            entry.update(status="complete", metrics_file=destination.name,
                         test_acc_mean=mean(r["test_accuracy"] for r in metrics),
                         test_bacc_mean=mean(r["test_bacc"] for r in metrics))
            finished[method][seed] = entry
        method_summary = {"runs": runs, "complete_runs": len(finished[method])}
        if len(finished[method]) == len(SEEDS):
            for metric in ("test_acc_mean", "test_bacc_mean"):
                values = [r[metric] for r in finished[method].values()]
                method_summary[metric] = {"mean": mean(values), "seed_sample_sd": stdev(values)}
        summary["methods"][method] = method_summary

    for control in ("off", "overall"):
        pair = {}
        for seed in SEEDS:
            if seed in finished["rival"] and seed in finished[control]:
                pair[seed] = {metric: finished["rival"][seed][metric] - finished[control][seed][metric]
                              for metric in ("test_acc_mean", "test_bacc_mean")}
        comparison = {"paired_seed_deltas_pp": pair}
        if len(pair) == len(SEEDS):
            acc = [r["test_acc_mean"] for r in pair.values()]
            bacc = [r["test_bacc_mean"] for r in pair.values()]
            comparison.update(acc_delta_mean_pp=mean(acc), bacc_delta_mean_pp=mean(bacc),
                              acc_delta_seed_sd_pp=stdev(acc),
                              meets_followup_threshold=min(acc) > 0 and mean(acc) >= .30 and mean(bacc) >= 0)
        summary["paired"][f"rival_minus_{control}"] = comparison

    shutil.copyfile(args.study_root / "outputs/margin_targets_logits_v2.json",
                    args.out / "targets_metadata.json")
    content = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    temporary = args.out / ".summary.json.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.out / "summary.json")
    print(content, end="")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--run-tag", default="formal")
    collect(p.parse_args())
