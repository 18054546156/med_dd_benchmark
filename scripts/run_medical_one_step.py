#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对每个算法执行一次最小真实计算闭环。

这里的 ``one-step`` 不是替代论文实验，而是验证医疗输入已经穿过
loader、网络、算法损失和反向更新；正式训练仍由各 raw 算法自己的入口负责。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SHARED_UTILS = ROOT / "utils"
DATASETS = ("PathMNIST", "COVID", "Kvasir")
ALGORITHMS = ("DC", "DSA", "DM", "MTT", "HoP-TM", "NCFM", "DataDAM", "CAFE")
SPECS = {
    "PathMNIST": (3, (32, 32), 9),
    "COVID": (3, (112, 112), 4),
    "Kvasir": (3, (128, 128), 8),
}


def load_module(path: Path, name: str, import_dir: Path):
    sys.path.insert(0, str(import_dir))
    sys.path.insert(0, str(SHARED_UTILS))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SHARED_UTILS))
        sys.path.remove(str(import_dir))


def one_per_class(dataset, num_classes):
    """优先使用 targets/labels 元数据，避免为 smoke 测试扫描全部图像。"""
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "labels", None)
    if labels is not None:
        labels = [int(torch.as_tensor(label).reshape(-1)[0]) for label in labels]
        selected = []
        for class_id in range(num_classes):
            selected.append(labels.index(class_id))
        return selected

    selected = {}
    for index in range(len(dataset)):
        label = int(torch.as_tensor(dataset[index][1]).reshape(-1)[0])
        selected.setdefault(label, index)
        if len(selected) == num_classes:
            break
    return [selected[class_id] for class_id in range(num_classes)]


def logits_from_output(output, batch_size, num_classes):
    """兼容不同官方网络的 tensor/tuple 输出。"""
    candidates = output if isinstance(output, (tuple, list)) else (output,)
    for candidate in candidates:
        if torch.is_tensor(candidate) and tuple(candidate.shape) == (batch_size, num_classes):
            return candidate
    raise AssertionError(f"找不到分类 logits: {[getattr(x, 'shape', None) for x in candidates]}")


def load_algorithm(algorithm, dataset, data_path):
    adapted = ROOT / "adapted"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if algorithm in {"DC", "DSA", "DM"}:
        module = load_module(adapted / "dc_dsa_dm" / "utils.py", "_step_dc", adapted / "dc_dsa_dm")
        result = module.get_dataset(dataset, str(data_path))
        return module, result[6], result[2], result[0], result[1], device
    if algorithm == "MTT":
        module = load_module(adapted / "mtt" / "utils.py", "_step_mtt", adapted / "mtt")
        args = SimpleNamespace(zca=False, device=device, workers=0)
        result = module.get_dataset(dataset, str(data_path), batch_size=2, args=args)
        return module, result[6], result[2], result[0], result[1], device
    if algorithm == "HoP-TM":
        module = load_module(
            adapted / "hop_tm" / "utils" / "utils_baseline.py",
            "_step_hop",
            adapted / "hop_tm",
        )
        args = SimpleNamespace(zca=False, device=device, workers=0)
        result = module.get_dataset(dataset, str(data_path), batch_size=2, args=args)
        return module, result[5], result[2], result[0], result[1], device
    if algorithm == "DataDAM":
        module = load_module(adapted / "datadam" / "utils.py", "_step_datadam", adapted / "datadam")
        args = SimpleNamespace(zca=False, device=device)
        result = module.get_dataset(dataset, str(data_path), args)
        return module, result[6], result[2], result[0], result[1], device
    if algorithm == "CAFE":
        module = load_module(adapted / "cafe" / "utils.py", "_step_cafe", adapted / "cafe")
        result = module.get_dataset(dataset, str(data_path))
        return module, result[6], result[2], result[0], result[1], device
    if algorithm == "NCFM":
        sys.path.insert(0, str(adapted / "ncfm"))
        sys.path.insert(0, str(SHARED_UTILS))
        try:
            from utils.utils import define_model, load_resized_data

            channel, im_size, num_classes = SPECS[dataset]
            train, _ = load_resized_data(
                dataset.lower(), str(data_path), size=im_size[0], nclass=num_classes
            )
            return (define_model, train, num_classes, channel, im_size, device)
        finally:
            sys.path.pop(0)
            sys.path.pop(0)
    raise ValueError(algorithm)


def run_one(algorithm, dataset, data_path):
    channel, im_size, expected_classes = SPECS[dataset]
    loaded = load_algorithm(algorithm, dataset, data_path)
    module, train, num_classes, channel, im_size, device = loaded
    if num_classes != expected_classes:
        raise AssertionError(f"类别数不一致: {num_classes} != {expected_classes}")

    indices = one_per_class(train, num_classes)
    images = torch.stack([train[index][0] for index in indices]).float().to(device)
    labels = torch.arange(num_classes, dtype=torch.long, device=device)

    if algorithm == "NCFM":
        model = module("medical", "instance", "convnet", channel, 3, 1.0, num_classes, None, im_size[0])
    else:
        model = module.get_network("ConvNet", channel, num_classes, im_size)
    model = model.to(device).train()

    # DataDAM/CAFE 的网络返回 tuple，其余官方 ConvNet 返回 logits tensor。
    if algorithm in {"DC", "DSA", "DM", "CAFE"}:
        criterion = nn.CrossEntropyLoss()
        synthetic = images.detach().clone().requires_grad_(True)
        real_logits = logits_from_output(model(images), num_classes, num_classes)
        real_grad = torch.autograd.grad(criterion(real_logits, labels), model.parameters())
        syn_logits = logits_from_output(model(synthetic), num_classes, num_classes)
        syn_grad = torch.autograd.grad(
            criterion(syn_logits, labels), model.parameters(), create_graph=True
        )
        args = SimpleNamespace(device=device, dis_metric="ours")
        loss = module.match_loss(syn_grad, [grad.detach() for grad in real_grad], args)
        loss.backward()
        if synthetic.grad is None:
            raise AssertionError("梯度匹配没有回传到合成图像")
        torch.optim.SGD([synthetic], lr=0.01).step()
        detail = f"gradient_match_loss={float(loss.detach()):.6f}"
    elif algorithm == "DataDAM":
        # DataDAM 官方网络的第二个返回值是分类 logits，第一个是 embedding。
        output = model(images)
        logits = logits_from_output(output, num_classes, num_classes)
        loss = nn.CrossEntropyLoss()(logits, labels)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        detail = f"classification_loss={float(loss.detach()):.6f}"
    elif algorithm == "NCFM":
        logits = logits_from_output(model(images), num_classes, num_classes)
        loss = nn.CrossEntropyLoss()(logits, labels)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        detail = f"classification_loss={float(loss.detach()):.6f}"
    elif algorithm in {"MTT", "HoP-TM"}:
        output = model(images)
        logits = logits_from_output(output, num_classes, num_classes)
        loss = nn.CrossEntropyLoss()(logits, labels)
        before = [parameter.detach().cpu().clone() for parameter in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after = [parameter.detach().cpu().clone() for parameter in model.parameters()]
        buffer_dir = ROOT / "buffers" / "smoke" / algorithm / dataset
        buffer_dir.mkdir(parents=True, exist_ok=True)
        torch.save([[before, after]], buffer_dir / "replay_buffer_0.pt")
        detail = f"trajectory_loss={float(loss.detach()):.6f}"
    else:
        raise AssertionError(algorithm)

    print(f"PASS {algorithm} {dataset} {detail}")


def main():
    parser = argparse.ArgumentParser(description="8x3 医疗 one-step 验证")
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "prepared")
    args = parser.parse_args()
    run_one(args.algorithm, args.dataset, args.data_path.resolve())


if __name__ == "__main__":
    main()
