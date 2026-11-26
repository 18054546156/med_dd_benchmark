"""Collect only completed metrics; never mistake smoke results for full runs."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--study-root", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--job-id", required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
summary = {"job_id":a.job_id,"training_steps":20000,"eval_epochs":2000,
           "expected_seeds":[0,1,2],"expected_repeats":5,"methods":{}}
for method in ("off","overall","rival"):
    rows=[]
    for seed in range(3):
        f=a.study_root/"outputs"/"formal"/method/f"seed{seed}"/"evaluation.json"
        if not f.exists():
            rows.append({"seed":seed,"status":"missing"})
            continue
        metrics=json.loads(f.read_text())
        if len(metrics)!=5:
            rows.append({"seed":seed,"status":"incomplete","repeat_count":len(metrics)})
            continue
        path=a.out/f"{method}_seed{seed}.json"
        shutil.copyfile(f,path)
        rows.append({"seed":seed,"status":"complete",
                     "test_acc_mean":float(np.mean([m["test_accuracy"] for m in metrics])),
                     "test_bacc_mean":float(np.mean([m["test_bacc"] for m in metrics])),
                     "metrics_file":path.name})
    complete=[r for r in rows if r["status"]=="complete"]
    summary["methods"][method]={"runs":rows,"complete_runs":len(complete),
        "mean_across_seeds":float(np.mean([r["test_acc_mean"] for r in complete])) if len(complete)==3 else None}
status=subprocess.run(["sacct","-j",a.job_id,"-n","-P","--format=JobID,State,Elapsed,NodeList"],
                      text=True,capture_output=True,check=False)
summary["sacct"]=status.stdout.splitlines()
shutil.copyfile(a.study_root/"outputs"/"margin_targets_logits_v2.json",a.out/"targets_metadata.json")
(a.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
