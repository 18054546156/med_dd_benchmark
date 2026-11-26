"""Write an isolated, fully specified PathMNIST experiment configuration."""
import argparse
from pathlib import Path
import yaml

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["off","overall","rival"], required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--teacher-dir", required=True)
p.add_argument("--out-root", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
config = yaml.safe_load((root/"configs/ncfm/pathmnist/ipc10_full.yaml").read_text(encoding="utf8"))
config["dataset"]["data_dir"] = a.data_root
config["dataset"]["batch_real"] = 128
config["train"]["seed"] = a.seed
config["save_path"]["pretrain_dir"] = a.teacher_dir
config["save_path"]["pretrain_dir_is_final"] = True
config["save_path"]["save_dir"] = str(a.out_root)
config["distibution_train"]["workers"] = 4
(a.out_root/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf8")
