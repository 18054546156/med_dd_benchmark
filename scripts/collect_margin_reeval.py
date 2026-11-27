"""Verify eval-only jobs, summarize paired results, and archive completed metrics."""
import argparse
import json
from pathlib import Path
import shutil
from statistics import mean, stdev

from collect_margin_results import MODES, SEEDS, read_metrics


def compare(a, b, fields):
    return {key: a[key] - b[key] for key in fields}


def run(args):
    accounting = args.accounting.read_text(encoding="utf-8")
    states = dict((p[0], p[1:]) for p in (s.split("|") for s in accounting.splitlines()))
    artifacts = [f"{mode}_seed{seed}" for mode in MODES for seed in SEEDS]
    tasks = [
        (f"{artifact}_{normalization}", f"{args.ab}_{index}")
        for index, (artifact, normalization) in enumerate(
            (a, n) for a in ("26479_0", "off_seed0") for n in ("raw", "once"))
    ]
    tasks += [(f"{artifact}_once", f"{args.reeval}_{index}")
              for index, artifact in enumerate(artifacts[1:], start=4)]
    overrides = dict(item.split("=", 1) for item in args.replace_task)
    if overrides.keys() - dict(tasks).keys():
        raise ValueError("unknown replacement case")
    args.out.mkdir(parents=True, exist_ok=True)
    probe_path = args.out / "margin_probe.json"
    low_classes = (json.loads(probe_path.read_text(encoding="utf-8"))["lowest_real_q10_classes"]
                   if probe_path.is_file() else [])
    summary = {"runs": {}, "ab": {}, "corrected": {}, "paired": {},
               "lowest_real_q10_classes": low_classes}
    verified, student_records = {}, {}
    metrics = ("test_accuracy", "test_bacc")
    for case, default_task in tasks:
        task = overrides.get(case, default_task)
        state, code, *info = states.get(task, ["UNKNOWN", ""])
        entry = {"task": task, "state": state, "exit_code": code, "accounting_details": info}
        summary["runs"][case] = entry
        if (state, code) != ("COMPLETED", "0:0"):
            continue
        source = args.results / case
        rows = read_metrics(source / "evaluation.json")
        destination = args.out / case
        destination.mkdir(exist_ok=True)
        for item in ("evaluation.json", "config.yaml", "synthetic_sha256.txt"):
            shutil.copyfile(source / item, destination / item)
        averages = {key: mean(row[key] for row in rows) for key in metrics}
        averages["class_recall_pct"] = [
            100 * mean(row["confusion"][c][c] / sum(row["confusion"][c]) for row in rows)
            for c in range(9)
        ]
        averages["class_accuracy_contribution_pct"] = [
            100 * mean(row["confusion"][c][c] / sum(map(sum, row["confusion"])) for row in rows)
            for c in range(9)
        ]
        entry.update(averages)
        verified[case], student_records[case] = averages, rows

    for artifact in ("26479_0", "off_seed0"):
        raw, once = f"{artifact}_raw", f"{artifact}_once"
        if {raw, once} <= verified.keys():
            summary["ab"][artifact] = {
                "delta_pp": compare(verified[once], verified[raw], metrics),
                "paired_repeat_deltas_pp": [compare(a, b, metrics)
                                            for a, b in zip(student_records[once],
                                                            student_records[raw])],
            }
    for mode in MODES:
        cases = [f"{mode}_seed{s}_once" for s in SEEDS]
        if set(cases) <= verified.keys():
            values = [[verified[c][name] for c in cases] for name in metrics]
            summary["corrected"][mode] = {name: {"mean": mean(v), "seed_sample_sd": stdev(v)}
                                          for name, v in zip(metrics, values)}
    for mode, control in (("overall", "off"), ("rival", "off"), ("rival", "overall")):
        deltas = []
        for seed in SEEDS:
            a, b = f"{mode}_seed{seed}_once", f"{control}_seed{seed}_once"
            if {a, b} <= verified.keys():
                row = compare(verified[a], verified[b], metrics)
                row["class_recall_pp"] = [x - y for x, y in zip(verified[a]["class_recall_pct"],
                                                               verified[b]["class_recall_pct"])]
                row["class_accuracy_contribution_pp"] = [
                    x - y for x, y in zip(verified[a]["class_accuracy_contribution_pct"],
                                          verified[b]["class_accuracy_contribution_pct"])
                ]
                deltas.append({"seed": seed, **row})
        if len(deltas) == len(SEEDS):
            record = {
                "mean_delta_pp": {name: mean(r[name] for r in deltas) for name in metrics},
                "paired_seed_deltas_pp": deltas,
            }
            if low_classes:
                others = sorted(set(range(9)) - set(low_classes))
                for label, classes in (("low_margin", low_classes), ("other", others)):
                    record[f"{label}_mean_recall_delta_pp"] = mean(
                        row["class_recall_pp"][c] for c in classes for row in deltas)
                    record[f"{label}_accuracy_delta_contribution_pp"] = mean(
                        sum(row["class_accuracy_contribution_pp"][c] for c in classes)
                        for row in deltas
                    )
            summary["paired"][f"{mode}_minus_{control}"] = record
    (args.out / "slurm_accounting.txt").write_text(accounting, encoding="utf-8")
    temporary = args.out / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.out / "summary.json")
    print(f"Verified {len(verified)}/{len(tasks)} cases")
    if args.require_complete and len(verified) != len(tasks):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accounting", "results", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--ab", required=True)
    parser.add_argument("--reeval", required=True)
    parser.add_argument("--replace-task", action="append", default=[], metavar="CASE=JOB_ID")
    parser.add_argument("--require-complete", action="store_true")
    run(parser.parse_args())
