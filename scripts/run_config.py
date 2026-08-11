#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按各算法原始入口执行一个医疗迁移 YAML。

这个脚本只负责配置解析、路径固定和入口分派，不把八个算法强行改成同一套
训练流程。默认只打印最终命令；加 ``--run`` 才会真正启动训练或 buffer 生成。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "prepared"
SPECS = {
    "PathMNIST": {"channel": 3, "im_size": [32, 32], "num_classes": 9},
    "COVID": {"channel": 3, "im_size": [112, 112], "num_classes": 4},
    "Kvasir": {"channel": 3, "im_size": [128, 128], "num_classes": 8},
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML 顶层必须是字典: {path}")
    return config


def cfg(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """读取扁平键或一层嵌套键，兼容不同算法的原始 YAML 结构。"""
    if key in config:
        return config[key]
    for section in ("buffer", "distillation", "network", "optimizer", "evaluation"):
        value = config.get(section)
        if isinstance(value, dict) and key in value:
            return value[key]
    return default


def add(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([f"--{name}", str(value)])


def add_bool(command: list[str], name: str, value: Any) -> None:
    if value:
        command.append(f"--{name}")


def add_smoke_overrides(command: list[str]) -> None:
    """把支持统一 smoke 的入口限制为一次最小计算闭环。"""
    for name, value in (("Iteration", 1), ("num_exp", 1),
                        ("num_eval", 1), ("epoch_eval_train", 1)):
        command.extend([f"--{name}", str(value)])


def output_path(config: dict[str, Any], algorithm: str, dataset: str) -> Path:
    raw = cfg(config, "save_path")
    if not raw:
        return ROOT / "results" / algorithm / dataset
    # NCFM 的 save_path 是字典，其它算法通常是字符串；统一入口只取 save_dir。
    if isinstance(raw, dict):
        raw = raw.get("save_dir")
    if not raw:
        return ROOT / "results" / algorithm / dataset
    value = str(raw).replace("\\", "/")
    # 配置常以算法子目录为当前目录；统一折叠到仓库根目录下。
    while value.startswith("../"):
        value = value[3:]
    while value.startswith("./"):
        value = value[2:]
    return (ROOT / value).resolve()


def validate_contract(config: dict[str, Any], dataset: str) -> None:
    expected = SPECS[dataset]
    section = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    actual_classes = config.get("num_classes", section.get("nclass"))
    actual_channel = config.get("channel", section.get("nch"))
    actual_size = config.get("im_size")
    if actual_size is None:
        size = section.get("size")
        actual_size = [size, size] if size is not None else None
    if actual_classes is not None and int(actual_classes) != expected["num_classes"]:
        raise ValueError(f"{dataset} num_classes 配置错误: {actual_classes}")
    if actual_channel is not None and int(actual_channel) != expected["channel"]:
        raise ValueError(f"{dataset} channel 配置错误: {actual_channel}")
    if actual_size is not None and list(actual_size) != expected["im_size"]:
        raise ValueError(f"{dataset} im_size 配置错误: {actual_size}")


def command_for(
    algorithm: str,
    stage: str,
    config: dict[str, Any],
    config_path: Path,
    load_path: str | None = None,
) -> tuple[list[str], Path]:
    dataset = config.get("dataset")
    if isinstance(dataset, dict):
        dataset = dataset.get("dataset")
    dataset_names = {"pathmnist": "PathMNIST", "covid": "COVID", "kvasir": "Kvasir"}
    dataset = dataset_names.get(str(dataset).lower(), dataset)
    if dataset not in SPECS:
        raise ValueError(f"配置中的 dataset 必须是 PathMNIST/COVID/Kvasir: {dataset}")
    validate_contract(config, dataset)
    data_path = DATA_ROOT
    save_path = output_path(config, algorithm, dataset)
    model = cfg(config, "model", "ConvNet")
    ipc = cfg(config, "ipc", cfg(config, "ipc", 10))

    if algorithm in {"dc", "dsa", "dm", "dc_dsa_dm"}:
        method = str(config.get("method", algorithm)).upper()
        if algorithm == "dc_dsa_dm":
            algorithm = method.lower()
        if algorithm == "dm" or method == "DM":
            script = ROOT / "adapted" / "dc_dsa_dm" / "main_DM.py"
        else:
            script = ROOT / "adapted" / "dc_dsa_dm" / "main.py"
        command = [sys.executable, str(script)]
        add(command, "method", method)
        for key in ("dataset", "model", "ipc", "eval_mode", "num_exp", "num_eval",
                    "epoch_eval_train", "Iteration", "lr_img", "lr_net",
                    "batch_real", "batch_train", "init", "dsa_strategy", "dis_metric"):
            add(command, key, dataset if key == "dataset" else cfg(config, key))
        add(command, "data_path", data_path)
        add(command, "save_path", save_path)
        add(command, "device", cfg(config, "device", "auto"))
        if stage == "smoke":
            add_smoke_overrides(command)
            command.append("--fast_eval")
        return command, script.parent

    if algorithm == "datadam":
        script = ROOT / "adapted" / "datadam" / "main_DataDAM.py"
        command = [sys.executable, str(script)]
        for key in ("dataset", "model", "ipc", "eval_mode", "num_exp", "num_eval",
                    "epoch_eval_train", "Iteration", "lr_img", "lr_net",
                    "batch_real", "batch_train", "init", "dsa_strategy", "task_balance"):
            add(command, key, dataset if key == "dataset" else cfg(config, key))
        add_bool(command, "zca", cfg(config, "zca", False))
        add(command, "data_path", data_path)
        add(command, "save_path", save_path)
        if stage == "smoke":
            add_smoke_overrides(command)
        return command, script.parent

    if algorithm == "cafe":
        script = ROOT / "adapted" / "cafe" / "distill.py"
        command = [sys.executable, str(script)]
        for key in ("dataset", "model", "ipc", "eval_mode", "num_exp", "num_eval",
                    "epoch_eval_train", "Iteration", "lr_img", "lr_net",
                    "batch_real", "batch_train", "init", "dsa_strategy",
                    "fourth_weight", "third_weight", "second_weight", "first_weight",
                    "inner_weight", "lambda_1", "lambda_2"):
            add(command, key, dataset if key == "dataset" else cfg(config, key))
        add(command, "data_path", data_path)
        add(command, "save_path", save_path)
        if stage == "smoke":
            add_smoke_overrides(command)
            command.append("--smoke")
        return command, script.parent

    if algorithm == "mtt":
        if stage == "buffer":
            script = ROOT / "adapted" / "mtt" / "buffer.py"
            section = config.get("buffer", {})
            command = [sys.executable, str(script)]
            for key in ("dataset", "model", "num_experts", "lr_teacher", "batch_train",
                        "batch_real", "dsa_strategy", "train_epochs", "save_interval"):
                value = dataset if key == "dataset" else section.get(key, cfg(config, key))
                add(command, key, value)
            add(command, "dsa", str(bool(config.get("augmentation", {}).get("dsa", True))))
            add(command, "data_path", data_path)
            add(command, "buffer_path", ROOT / "buffers" / "mtt")
            add_bool(command, "zca", cfg(config, "zca", False))
        else:
            script = ROOT / "adapted" / "mtt" / "distill.py"
            section = config.get("distillation", {})
            command = [sys.executable, str(script)]
            for key in ("dataset", "model", "ipc", "eval_mode", "num_eval", "eval_it",
                        "epoch_eval_train", "Iteration", "lr_img", "lr_lr", "lr_teacher",
                        "batch_real", "batch_syn", "batch_train", "expert_epochs",
                        "syn_steps", "max_start_epoch", "pix_init", "dsa_strategy"):
                value = dataset if key == "dataset" else section.get(key, cfg(config, key))
                add(command, key, value)
            add(command, "dsa", str(bool(config.get("augmentation", {}).get("dsa", True))))
            add(command, "data_path", data_path)
            add(command, "buffer_path", ROOT / "buffers" / "mtt")
            add(command, "save_path", save_path)
        return command, script.parent

    if algorithm == "hop_tm":
        if stage == "buffer":
            script = ROOT / "adapted" / "hop_tm" / "buffer" / "buffer_FTD.py"
            command = [sys.executable, str(script)]
            for key in ("dataset", "model", "num_experts", "lr_teacher", "batch_train",
                        "batch_real", "dsa_strategy", "train_epochs", "save_interval",
                        "mom", "l2", "rho_max", "rho_min", "alpha", "adaptive"):
                add(command, key, dataset if key == "dataset" else cfg(config, key))
            add(command, "dsa", str(bool(config.get("dsa", True))))
            add(command, "data_path", data_path)
            add(command, "buffer_path", ROOT / "buffers" / "hop_tm")
            add_bool(command, "zca", cfg(config, "zca", False))
        else:
            script = ROOT / "adapted" / "hop_tm" / "distill" / "distill_high_order_spl.py"
            command = [sys.executable, str(script), "--cfg", str(config_path.resolve())]
            add(command, "data_path", data_path)
            add(command, "buffer_path", ROOT / "buffers" / "hop_tm")
            add(command, "save_path", save_path)
        return command, script.parent.parent if stage == "distill" else script.parent.parent

    if algorithm == "ncfm":
        stage_script = {
            "pretrain": ROOT / "adapted" / "ncfm" / "pretrain" / "pretrain_script.py",
            "condense": ROOT / "adapted" / "ncfm" / "condense" / "condense_script.py",
            "evaluation": ROOT / "adapted" / "ncfm" / "evaluation" / "evaluation_script.py",
        }.get(stage)
        if stage_script is None:
            raise ValueError("NCFM stage 必须是 pretrain/condense/evaluation")
        # 三个 NCFM 脚本虽然都读取同一份 YAML，但 run_mode 不同；显式传入，
        # 避免把 pretrain、condense、evaluation 错当成同一个入口。
        run_modes = {"pretrain": "Pretrain", "condense": "Condense", "evaluation": "Evaluation"}
        command = [sys.executable, str(stage_script), "--config_path", str(config_path.resolve()),
                   "--run_mode", run_modes[stage], "--gpu", "0"]
        add(command, "ipc", cfg(config, "ipc", cfg(config, "condense", {}).get("ipc", 1)))
        if stage == "evaluation":
            resolved_load_path = load_path or config.get("load_path")
            if not resolved_load_path:
                raise ValueError("NCFM evaluation 需要显式 --load-path，避免误用错误的合成数据")
            add(command, "load_path", Path(resolved_load_path).resolve())
        return command, ROOT / "adapted" / "ncfm"

    raise ValueError(f"不支持的算法: {algorithm}")


def main() -> int:
    parser = argparse.ArgumentParser(description="按算法原始入口执行医疗迁移配置")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["dc", "dsa", "dm", "dc_dsa_dm", "mtt", "hop_tm", "ncfm", "datadam", "cafe"])
    parser.add_argument("--stage", default="distill", choices=["distill", "buffer", "pretrain", "condense", "evaluation", "smoke"])
    parser.add_argument("--load-path", help="NCFM evaluation 使用的合成数据目录或文件")
    parser.add_argument("--run", action="store_true", help="真正启动命令；默认只打印命令")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    config = load_config(config_path.resolve())
    algorithm = args.algorithm or config_path.parts[-3]
    if algorithm == "dc_dsa_dm" and config.get("method"):
        algorithm = str(config["method"]).lower()
    command, cwd = command_for(algorithm, args.stage, config, config_path.resolve(), args.load_path)
    print("配置:", config_path.resolve())
    print("工作目录:", cwd)
    print("命令:", subprocess.list2cmdline([str(x) for x in command]))
    if args.run:
        return subprocess.run(command, cwd=cwd).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
