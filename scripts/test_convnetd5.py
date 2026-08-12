#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证八个算法在三个医疗数据集上的网络构造和真实前向传播。

本脚本只验证模型级闭环，不代表完整蒸馏训练已经完成。每个算法在独立
子进程中导入，避免不同原始仓库的同名 ``utils`` 模块互相污染。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_SPECS = {
    "PathMNIST": {"model": "ConvNet", "size": (32, 32), "classes": 9},
    "COVID": {"model": "ConvNetD5", "size": (112, 112), "classes": 4},
    "Kvasir": {"model": "ConvNetD5", "size": (128, 128), "classes": 8},
}

ALGORITHMS = {
    "DC": {
        "cwd": PROJECT_ROOT / "adapted" / "dc_dsa_dm",
        "import_code": "from utils import get_network",
        "factory": "get_network(model, channel=3, num_classes=classes, im_size=size)",
    },
    "DSA": {
        "cwd": PROJECT_ROOT / "adapted" / "dc_dsa_dm",
        "import_code": "from utils import get_network",
        "factory": "get_network(model, channel=3, num_classes=classes, im_size=size)",
    },
    "DM": {
        "cwd": PROJECT_ROOT / "adapted" / "dc_dsa_dm",
        "import_code": "from utils import get_network",
        "factory": "get_network(model, channel=3, num_classes=classes, im_size=size)",
    },
    "MTT": {
        "cwd": PROJECT_ROOT / "adapted" / "mtt",
        "import_code": "from utils import get_network",
        "factory": "get_network(model, channel=3, num_classes=classes, im_size=size, dist=False)",
    },
    "HoP-TM": {
        "cwd": PROJECT_ROOT / "adapted" / "hop_tm",
        "import_code": "from utils.utils_gsam import get_network",
        "factory": "get_network(model, channel=3, num_classes=classes, im_size=size)",
    },
    "DataDAM": {
        "cwd": PROJECT_ROOT / "adapted" / "datadam",
        "import_code": "from utils import get_network",
        "factory": "get_network(model, channel=3, num_classes=classes, im_size=size)",
    },
    "CAFE": {
        "cwd": PROJECT_ROOT / "adapted" / "cafe",
        "import_code": "from utils import get_network",
        "factory": "get_network(model, channel=3, num_classes=classes, im_size=size)",
    },
    "NCFM": {
        "cwd": PROJECT_ROOT / "adapted" / "ncfm",
        "import_code": (
            "from utils.utils import define_model\n"
            "model = None"
        ),
        "factory": (
            "define_model(dataset.lower(), 'instance', 'convnet', 3, depth, "
            "1.0, classes, None, size_value)"
        ),
    },
}


def make_probe(import_code: str, factory: str, dataset: str, spec: dict) -> str:
    """生成一个只使用随机输入的独立模型前向测试程序。"""
    height, width = spec["size"]
    classes = spec["classes"]
    model = spec["model"]
    depth = 3 if model == "ConvNet" else 5
    # NCFM 的 define_model 接收单个整数 size；其它原始仓库接收 (H, W)。
    size_value = height
    return f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
{import_code}

dataset = {dataset!r}
model = {model!r}
size = ({height}, {width})
classes = {classes}
depth = {depth}
size_value = {height}
net = {factory}.cpu()
net.eval()
with torch.no_grad():
    output = net(torch.randn(2, 3, {height}, {width}))

candidates = output if isinstance(output, (tuple, list)) else (output,)
logits = next(
    value for value in candidates
    if torch.is_tensor(value) and tuple(value.shape) == (2, classes)
)
assert tuple(logits.shape) == (2, classes)
print(f"PARAMS={{sum(p.numel() for p in net.parameters())}} OUTPUT={{tuple(logits.shape)}}")
'''


def run_case(algorithm: str, config: dict, dataset: str) -> tuple[bool, str]:
    spec = DATASET_SPECS[dataset]
    probe = make_probe(config["import_code"], config["factory"], dataset, spec)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=config["cwd"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def main() -> int:
    print("=" * 78)
    print("医疗 DD 网络模型真实前向测试（不是完整蒸馏训练）")
    print("=" * 78)
    import argparse

    parser = argparse.ArgumentParser(description="验证 DD backbone 的真实前向传播")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS),
                        help="只验证一个算法；不指定则验证全部算法")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS),
                        help="只验证一个数据集；不指定则验证全部数据集")
    args = parser.parse_args()

    # 单项验证必须真的缩小范围，否则一个 24 项测试会被误报成单项超时。
    algorithms = [args.algorithm] if args.algorithm else list(ALGORITHMS)
    datasets = [args.dataset] if args.dataset else list(DATASET_SPECS)
    failures = []

    for algorithm in algorithms:
        config = ALGORITHMS[algorithm]
        for dataset in datasets:
            passed, output = run_case(algorithm, config, dataset)
            if passed:
                print(f"PASS {algorithm:12s} {dataset:9s} {output}")
            else:
                print(f"FAIL {algorithm:12s} {dataset:9s}\n{output}")
                failures.append((algorithm, dataset))

    total = len(algorithms) * len(datasets)
    print(f"SUMMARY passed={total - len(failures)}/{total} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
