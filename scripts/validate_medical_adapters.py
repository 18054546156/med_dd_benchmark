#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证八个 DD 入口在三个医疗数据集上的真实 loader 合同。

每个算法在独立 Python 子进程中加载，避免不同官方仓库的 ``utils`` 和
``networks`` 模块互相复用。这个脚本验证 loader、标签、尺寸和算法特有
返回值；它不把 loader 验证冒充成完整蒸馏训练。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SHARED_UTILS = ROOT / "utils"
DATASETS = ("PathMNIST", "COVID", "Kvasir")
ALGORITHMS = (
    "DC",
    "DSA",
    "DM",
    "MTT",
    "HoP-TM",
    "NCFM",
    "DataDAM",
    "CAFE",
)
SPECS = {
    "PathMNIST": ((3, 32, 32), 9),
    "COVID": ((3, 112, 112), 4),
    "Kvasir": ((3, 128, 128), 8),
}


def load_module(module_path: Path, module_name: str, import_dir: Path):
    """加载一个官方适配模块，并将其算法目录放到导入优先级首位。"""
    sys.path.insert(0, str(import_dir))
    sys.path.insert(0, str(SHARED_UTILS))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SHARED_UTILS))
        sys.path.remove(str(import_dir))


def inspect_dataset(dataset_name: str, train_set, test_set, num_classes: int) -> None:
    """检查一个算法返回的训练集和测试集。"""
    expected_shape, expected_classes = SPECS[dataset_name]
    if num_classes != expected_classes:
        raise AssertionError(f"类别数错误: {num_classes} != {expected_classes}")
    image, label = train_set[0]
    if tuple(image.shape) != expected_shape:
        raise AssertionError(f"图像尺寸错误: {tuple(image.shape)} != {expected_shape}")
    if not isinstance(label, int):
        raise AssertionError(f"单样本标签不是 Python int: {type(label)!r}")
    loader = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    if tuple(images.shape[1:]) != expected_shape:
        raise AssertionError(f"batch 图像尺寸错误: {tuple(images.shape[1:])}")
    if tuple(labels.shape) != (2,) or labels.dtype != torch.long:
        raise AssertionError(f"batch 标签错误: shape={tuple(labels.shape)}, dtype={labels.dtype}")
    if int(labels.min()) < 0 or int(labels.max()) >= expected_classes:
        raise AssertionError(f"标签越界: {labels.tolist()}")


def run_one(algorithm: str, dataset: str, data_path: Path) -> None:
    """在当前隔离进程中执行一个算法的数据入口验证。"""
    adapted = ROOT / "adapted"
    if algorithm in {"DC", "DSA", "DM"}:
        module = load_module(adapted / "dc_dsa_dm" / "utils.py", "_dc_utils", adapted / "dc_dsa_dm")
        result = module.get_dataset(dataset, str(data_path))
        inspect_dataset(dataset, result[6], result[7], result[2])
    elif algorithm == "MTT":
        module = load_module(adapted / "mtt" / "utils.py", "_mtt_utils", adapted / "mtt")
        args = SimpleNamespace(zca=False, device="cuda" if torch.cuda.is_available() else "cpu", workers=0)
        result = module.get_dataset(dataset, str(data_path), batch_size=2, args=args)
        inspect_dataset(dataset, result[6], result[7], result[2])
        loader_train_dict = result[9]
        if set(loader_train_dict) != set(range(result[2])):
            raise AssertionError(f"MTT loader_train_dict 类别错误: {sorted(loader_train_dict)}")
    elif algorithm == "HoP-TM":
        module = load_module(
            adapted / "hop_tm" / "utils" / "utils_baseline.py",
            "_hop_baseline_utils",
            adapted / "hop_tm",
        )
        args = SimpleNamespace(zca=False, device="cuda" if torch.cuda.is_available() else "cpu", workers=0)
        result = module.get_dataset(dataset, str(data_path), batch_size=2, args=args)
        inspect_dataset(dataset, result[5], result[6], result[2])
    elif algorithm == "NCFM":
        sys.path.insert(0, str(adapted / "ncfm"))
        sys.path.insert(0, str(SHARED_UTILS))
        try:
            from utils.utils import load_resized_data

            size = SPECS[dataset][0][-1]
            result = load_resized_data(dataset.lower(), str(data_path), size=size, nclass=SPECS[dataset][1])
        finally:
            sys.path.pop(0)
            sys.path.pop(0)
        inspect_dataset(dataset, result[0], result[1], result[0].nclass)
    elif algorithm == "DataDAM":
        module = load_module(adapted / "datadam" / "utils.py", "_datadam_utils", adapted / "datadam")
        args = SimpleNamespace(zca=False, device="cuda" if torch.cuda.is_available() else "cpu")
        result = module.get_dataset(dataset, str(data_path), args)
        inspect_dataset(dataset, result[6], result[7], result[2])
    elif algorithm == "CAFE":
        module = load_module(adapted / "cafe" / "utils.py", "_cafe_utils", adapted / "cafe")
        result = module.get_dataset(dataset, str(data_path))
        inspect_dataset(dataset, result[6], result[7], result[2])
    else:
        raise ValueError(f"未知算法: {algorithm}")
    print(f"PASS {algorithm} {dataset}")


def main() -> int:
    parser = argparse.ArgumentParser(description="8x3 医疗数据 loader 验证")
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "prepared")
    args = parser.parse_args()

    if args.algorithm and args.dataset:
        run_one(args.algorithm, args.dataset, args.data_path.resolve())
        return 0

    failures = []
    for algorithm in ALGORITHMS:
        for dataset in DATASETS:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--algorithm",
                algorithm,
                "--dataset",
                dataset,
                "--data-path",
                str(args.data_path.resolve()),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            output = (result.stdout + result.stderr).strip()
            print(output[-1200:])
            if result.returncode:
                failures.append((algorithm, dataset, output[-500:]))

    print(f"SUMMARY passed={len(ALGORITHMS) * len(DATASETS) - len(failures)}/24 failed={len(failures)}")
    for algorithm, dataset, error in failures:
        print(f"FAIL {algorithm} {dataset}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
