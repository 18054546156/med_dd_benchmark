#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按各算法原始入口执行一个医疗迁移 YAML。

这个脚本只负责配置解析、路径固定和入口分派，不把八个算法强行改成同一套
训练流程。默认只打印最终命令；加 ``--run`` 才会真正启动训练或 buffer 生成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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


def dataset_from_config(config: dict[str, Any]) -> str:
    """把扁平/嵌套 YAML 中的数据集名称统一成目录合同名称。"""
    dataset = config.get("dataset")
    if isinstance(dataset, dict):
        dataset = dataset.get("dataset")
    names = {"pathmnist": "PathMNIST", "covid": "COVID", "kvasir": "Kvasir"}
    return names.get(str(dataset).lower(), str(dataset))


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


def add_smoke_overrides(command: list[str], include_fast_eval: bool = False) -> None:
    """把支持统一 smoke 的入口限制为一次最小计算闭环。"""
    for name, value in (("Iteration", 1), ("num_exp", 1),
                        ("num_eval", 1), ("epoch_eval_train", 1)):
        command.extend([f"--{name}", str(value)])
    if include_fast_eval:
        command.append("--fast_eval")


def _sha256(path: Path) -> str | None:
    """计算文件 SHA256；数据 manifest 不存在时返回 None。"""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    """读取 Git 信息；代码目录不是 Git 仓库时不阻断实验。"""
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _run_dir(algorithm: str, dataset: str, stage: str) -> Path:
    """为每次真实运行建立独立日志目录，避免不同参数相互覆盖。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "logs" / algorithm / dataset / stage / timestamp


def _write_run_manifest(
    run_dir: Path,
    *,
    config_path: Path,
    algorithm: str,
    dataset: str,
    stage: str,
    command: list[str],
    cwd: Path,
    start_time: str,
    end_time: str | None,
    return_code: int | None,
    status: str,
) -> None:
    """写入运行级 manifest；它记录实际命令，不把 YAML 元数据当成生效参数。"""
    data_manifest = DATA_ROOT / dataset / "manifest.json"
    payload = {
        "algorithm": algorithm,
        "dataset": dataset,
        "stage": stage,
        "status": status,
        "command": [str(item) for item in command],
        "working_directory": str(cwd),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data_manifest_path": str(data_manifest) if data_manifest.exists() else None,
        "data_manifest_sha256": _sha256(data_manifest),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--porcelain"),
        "start_time": start_time,
        "end_time": end_time,
        "return_code": return_code,
        "python": sys.executable,
        "pid": os.getpid(),
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def resource_path(config: dict[str, Any], key: str, default: Path) -> Path:
    """解析 buffer/checkpoint 等资源路径，统一相对项目根目录。

    MTT 和 HoP-TM 的 buffer 必须由当前 YAML 明确指定；运行器不能
    擅自把不同协议的 buffer 都重定向到同一个目录。
    """
    raw = cfg(config, key)
    if raw is None or str(raw).strip() == "":
        return default.resolve()
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_ncfm_load_path(raw_path: str | os.PathLike[str]) -> Path:
    """解析 NCFM evaluation 的合成数据路径。

    NCFM 原始 evaluation 入口要求 ``--load_path`` 指向具体的 ``.pt`` 文件，
    而 condense 阶段通常把多个 ``data_*.pt`` 文件放在 ``distilled_data`` 目录。
    统一运行器允许用户传目录，但在真正启动原始入口前必须解析成明确文件，
    否则会把一个目录传给 ``torch.load`` 并在运行中才失败。
    """
    path = Path(raw_path).expanduser()
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    if path.is_file():
        if path.suffix.lower() != ".pt":
            raise ValueError(f"NCFM evaluation 的 load-path 必须是 .pt 文件: {path}")
        return path

    if not path.exists():
        raise FileNotFoundError(f"NCFM evaluation 的 load-path 不存在: {path}")
    if not path.is_dir():
        raise ValueError(f"NCFM evaluation 的 load-path 不是文件或目录: {path}")

    candidates = list(path.rglob("data_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"NCFM evaluation 目录中没有 data_*.pt 合成数据: {path}"
        )

    # 完整 condense 会生成 data_1000.pt、data_2000.pt 等文件；正式评估
    # 必须优先选择数字最大的最终迭代文件，不能误评估 data_init.pt。
    numeric_candidates = []
    for candidate in candidates:
        suffix = candidate.stem.removeprefix("data_")
        try:
            numeric_candidates.append((int(suffix), candidate))
        except ValueError:
            continue
    if numeric_candidates:
        return max(numeric_candidates, key=lambda item: item[0])[1]

    # smoke 可能只有 data_init.pt；没有数字迭代文件时再回退到它。
    initial = path / "data_init.pt"
    if initial.is_file():
        return initial

    # 兼容非标准文件名，最后按修改时间兜底。
    def iteration_key(candidate: Path) -> tuple[int, float]:
        stem = candidate.stem
        suffix = stem.removeprefix("data_")
        try:
            return (int(suffix), candidate.stat().st_mtime)
        except ValueError:
            return (-1, candidate.stat().st_mtime)

    return max(candidates, key=iteration_key)


def require_stage(algorithm: str, stage: str, allowed: set[str]) -> None:
    """拒绝把某算法的阶段误当成另一个阶段执行。"""
    if stage not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(
            f"{algorithm} 不支持 stage={stage!r}；允许阶段为: {allowed_text}。"
        )


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
    dataset = dataset_from_config(config)
    if dataset not in SPECS:
        raise ValueError(f"配置中的 dataset 必须是 PathMNIST/COVID/Kvasir: {dataset}")
    validate_contract(config, dataset)
    data_path = DATA_ROOT
    save_path = output_path(config, algorithm, dataset)
    model = cfg(config, "model", "ConvNet")
    ipc = cfg(config, "ipc", cfg(config, "ipc", 10))

    if algorithm in {"dc", "dsa", "dm", "dc_dsa_dm"}:
        require_stage(algorithm, stage, {"distill", "smoke"})
        method = str(config.get("method", algorithm)).upper()
        if algorithm == "dc_dsa_dm":
            algorithm = method.lower()
        if algorithm == "dm" or method == "DM":
            script = ROOT / "adapted" / "dc_dsa_dm" / "main_DM.py"
        else:
            script = ROOT / "adapted" / "dc_dsa_dm" / "main.py"
        command = [sys.executable, str(script)]
        # main.py 解析 --method；main_DM.py 使用固定的 DM 入口，不接受该参数。
        if script.name == "main.py":
            add(command, "method", method)
        for key in ("dataset", "model", "ipc", "eval_mode", "num_exp", "num_eval",
                    "epoch_eval_train", "Iteration", "lr_img", "lr_net",
                    "batch_real", "batch_train", "init", "dsa_strategy", "dis_metric"):
            add(command, key, dataset if key == "dataset" else cfg(config, key))
        add(command, "data_path", data_path)
        add(command, "save_path", save_path)
        add(command, "device", cfg(config, "device", "auto"))
        if stage == "smoke":
            # 只有 main.py 声明了 --fast_eval；main_DM.py 没有这个参数。
            add_smoke_overrides(command, include_fast_eval=script.name == "main.py")
        return command, script.parent

    if algorithm == "datadam":
        require_stage(algorithm, stage, {"distill", "smoke"})
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
        require_stage(algorithm, stage, {"distill", "smoke"})
        script = ROOT / "adapted" / "cafe" / "distill.py"
        command = [sys.executable, str(script)]
        # CAFE 的原始 run.sh 不传 method，入口默认 DC（CAFE 的 feature
        # alignment 仍由本文件实现）。method=CAFE 是 YAML 标识字段，不能
        # 传入后误触发其它增强分支。
        add(command, "method", "DC")
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
        require_stage(algorithm, stage, {"buffer", "distill"})
        # MTT 的 buffer 和 distill 必须使用同一个网络结构；否则轨迹参数形状不兼容。
        buffer_section = config.get("buffer", {})
        buffer_model = buffer_section.get("model", model)
        if str(buffer_model) != str(model):
            raise ValueError(
                f"MTT model 不一致：顶层 model={model}，buffer.model={buffer_model}。"
            )
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
            add(command, "buffer_path", resource_path(
                config, "buffer_path", ROOT / "buffers" / "mtt"
            ))
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
            buffer_path = resource_path(
                config, "buffer_path", ROOT / "buffers" / "mtt"
            )
            if stage == "distill":
                expected = buffer_path / dataset
                if not cfg(config, "zca", False):
                    expected = Path(f"{expected}_NO_ZCA")
                expected = expected / str(model)
                if not expected.exists():
                    raise FileNotFoundError(
                        f"MTT distill 需要当前配置对应的 expert buffer，未找到: {expected}。"
                    )
            add(command, "buffer_path", buffer_path)
            add(command, "save_path", save_path)
        return command, script.parent

    if algorithm == "hop_tm":
        require_stage(algorithm, stage, {"buffer", "distill"})
        if stage == "buffer":
            script = ROOT / "adapted" / "hop_tm" / "buffer" / "buffer_FTD.py"
            command = [sys.executable, str(script)]
            for key in ("dataset", "model", "num_experts", "lr_teacher", "batch_train",
                        "batch_real", "dsa_strategy", "train_epochs", "save_interval",
                        "mom", "l2", "rho_max", "rho_min", "alpha", "adaptive"):
                add(command, key, dataset if key == "dataset" else cfg(config, key))
            add(command, "dsa", str(bool(config.get("dsa", True))))
            add(command, "data_path", data_path)
            add(command, "buffer_path", resource_path(
                config, "buffer_path", ROOT / "buffers" / "hop_tm"
            ))
            add_bool(command, "zca", cfg(config, "zca", False))
        else:
            script = ROOT / "adapted" / "hop_tm" / "distill" / "distill_high_order_spl.py"
            command = [sys.executable, str(script), "--cfg", str(config_path.resolve())]
            add(command, "data_path", data_path)
            buffer_path = resource_path(
                config, "buffer_path", ROOT / "buffers" / "hop_tm"
            )
            if stage == "distill":
                expected = buffer_path / dataset
                if not cfg(config, "zca", False):
                    expected = Path(f"{expected}_NO_ZCA")
                expected = expected / str(model)
                if not expected.exists():
                    raise FileNotFoundError(
                        f"HoP-TM distill 需要当前配置对应的 expert buffer，未找到: {expected}。"
                    )
            add(command, "buffer_path", buffer_path)
            add(command, "save_path", save_path)
        return command, script.parent.parent if stage == "distill" else script.parent.parent

    if algorithm == "ncfm":
        require_stage(algorithm, stage, {"pretrain", "condense", "evaluation"})
        if stage == "smoke":
            raise ValueError(
                "NCFM 不能用单条 smoke 命令代替完整流程；请依次运行 "
                "--stage pretrain、--stage condense、--stage evaluation。"
            )
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
            # NCFM 原始入口的 val_repeat 不在主 YAML 结构中时使用作者默认值 10；
            # smoke 配置可显式设为 1，避免把最小闭环误跑成十次完整评估。
            add(command, "val_repeat", config.get("val_repeat", 10))
            resolved_load_path = load_path or config.get("load_path")
            if not resolved_load_path:
                raise ValueError("NCFM evaluation 需要显式 --load-path，避免误用错误的合成数据")
            add(command, "load_path", resolve_ncfm_load_path(resolved_load_path))
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
        dataset_name = dataset_from_config(config)
        run_dir = _run_dir(algorithm, dataset_name, args.stage)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path.resolve(), run_dir / "config.yaml")
        (run_dir / "command.txt").write_text(
            subprocess.list2cmdline([str(x) for x in command]), encoding="utf-8"
        )
        start_time = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(
            run_dir,
            config_path=config_path.resolve(),
            algorithm=algorithm,
            dataset=dataset_name,
            stage=args.stage,
            command=command,
            cwd=cwd,
            start_time=start_time,
            end_time=None,
            return_code=None,
            status="running",
        )
        log_path = run_dir / "stdout.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
            return_code = process.wait()
        _write_run_manifest(
            run_dir,
            config_path=config_path.resolve(),
            algorithm=algorithm,
            dataset=dataset_name,
            stage=args.stage,
            command=command,
            cwd=cwd,
            start_time=start_time,
            end_time=datetime.now(timezone.utc).isoformat(),
            return_code=return_code,
            status="success" if return_code == 0 else "failed",
        )
        print(f"运行 manifest: {run_dir / 'run_manifest.json'}")
        return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
