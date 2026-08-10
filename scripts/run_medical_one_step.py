#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对每个算法执行一次最小真实计算闭环。

这里的 ``one-step`` 不是替代论文实验：匹配类算法调用其核心 loss，
MTT/HoP-TM/NCFM 做快速网络或轨迹 probe；完整训练仍由各适配入口负责。
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


def freeze_parameters(model):
    """冻结教师网络参数，但保留输入图像的梯度链。"""
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def make_synthetic(images):
    """从真实样本附近初始化合成样本，确保 smoke 能观察到非零更新。"""
    return (
        images.detach().clone() + 1e-3 * torch.randn_like(images)
    ).requires_grad_(True)


def require_input_gradient(synthetic):
    """确认匹配损失确实把梯度传回合成图像。"""
    if synthetic.grad is None:
        raise AssertionError("算法匹配损失没有回传到合成图像")


def run_distribution_matching(model, images, synthetic):
    """执行 DM 的 embedding 均值匹配，而不是复用 DC 的梯度匹配。"""
    freeze_parameters(model)
    embed = model.module.embed if torch.cuda.device_count() > 1 else model.embed
    real_feature = embed(images).detach()
    syn_feature = embed(synthetic)
    loss = torch.sum((real_feature.mean(dim=0) - syn_feature.mean(dim=0)) ** 2)
    loss.backward()
    return loss


def run_cafe_matching(module, model, images, synthetic, labels, num_classes):
    """执行 CAFE 的多层 feature alignment 与 inner output matching。"""
    freeze_parameters(model)
    real_output, real_features = model(images)
    syn_output, syn_features = model(synthetic)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(real_output, labels)
    feature_weights = ((-1, 0.1), (-2, 0.1), (-3, 1.0), (-4, 1.0))

    def criterion_middle(real_feature, syn_feature):
        real_shape = real_feature.shape
        real_mean = real_feature.reshape(
            num_classes, real_shape[0] // num_classes, *real_shape[1:]
        ).mean(dim=1)
        syn_shape = syn_feature.shape
        syn_mean = syn_feature.reshape(
            num_classes, syn_shape[0] // num_classes, *syn_shape[1:]
        ).mean(dim=1)
        return nn.MSELoss(reduction="sum")(real_mean, syn_mean)

    for index, weight in feature_weights:
        real_feature = real_features[index]
        syn_feature = syn_features[index]
        loss = loss + weight * criterion_middle(real_feature, syn_feature)

    # CAFE 的第一层特征同时参与类中心和 inner-loop 分类匹配。
    real_first = real_features[0].reshape(
        num_classes, -1, *real_features[0].shape[1:]
    ).mean(dim=1)
    syn_first = syn_features[0].reshape(
        num_classes, -1, *syn_features[0].shape[1:]
    ).mean(dim=1)
    class_logits = torch.mm(real_features[0], syn_first.t())
    loss = loss + criterion_middle(syn_first, real_first)
    loss = loss + 0.01 * nn.CrossEntropyLoss(reduction="sum")(class_logits, labels)
    loss.backward()
    return loss


def run_datadam_matching(module, model, images, synthetic, num_classes):
    """执行 DataDAM 的 ReLU attention/output matching。"""
    freeze_parameters(model)
    activations = {}

    def capture(name):
        def hook(_, __, output):
            activations[name] = output.clone()

        return hook

    base = model.module if torch.cuda.device_count() > 1 else model
    hooks = []
    for name, layer in base.features.named_modules():
        if isinstance(layer, nn.ReLU):
            hooks.append(layer.register_forward_hook(capture(f"ReLU_{len(hooks)}")))

    real_output = model(images)[0].detach()
    real_activations = list(activations.values())
    activations.clear()
    syn_output = model(synthetic)[0]
    syn_activations = list(activations.values())
    for hook in hooks:
        hook.remove()

    def batch_error(real, syn):
        real_mean = real.reshape(num_classes, -1).mean(dim=1)
        syn_mean = syn.reshape(num_classes, -1).mean(dim=1)
        return torch.sum((real_mean - syn_mean) ** 2)

    loss = torch.zeros((), device=synthetic.device)
    for real_activation, syn_activation in zip(
        real_activations[:-1], syn_activations[:-1]
    ):
        real_attention = module.get_attention(real_activation.detach(), param=1, exp=1, norm="l2")
        syn_attention = module.get_attention(syn_activation, param=1, exp=1, norm="l2")
        loss = loss + 100.0 * batch_error(real_attention, syn_attention)
    loss = loss + 100.0 * 0.01 * batch_error(real_output, syn_output)
    loss.backward()
    return loss


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
    if algorithm in {"DC", "DSA"}:
        criterion = nn.CrossEntropyLoss()
        synthetic = make_synthetic(images)
        real_logits = logits_from_output(model(images), num_classes, num_classes)
        real_grad = torch.autograd.grad(criterion(real_logits, labels), model.parameters())
        syn_logits = logits_from_output(model(synthetic), num_classes, num_classes)
        syn_grad = torch.autograd.grad(
            criterion(syn_logits, labels), model.parameters(), create_graph=True
        )
        args = SimpleNamespace(device=device, dis_metric="ours")
        loss = module.match_loss(syn_grad, [grad.detach() for grad in real_grad], args)
        loss.backward()
        require_input_gradient(synthetic)
        torch.optim.SGD([synthetic], lr=0.01).step()
        detail = f"gradient_match_loss={float(loss.detach()):.6f}"
    elif algorithm == "DM":
        synthetic = make_synthetic(images)
        loss = run_distribution_matching(model, images, synthetic)
        require_input_gradient(synthetic)
        torch.optim.SGD([synthetic], lr=0.01).step()
        detail = f"distribution_match_loss={float(loss.detach()):.6f}"
    elif algorithm == "CAFE":
        synthetic = make_synthetic(images)
        loss = run_cafe_matching(module, model, images, synthetic, labels, num_classes)
        require_input_gradient(synthetic)
        torch.optim.SGD([synthetic], lr=0.01).step()
        detail = f"feature_match_loss={float(loss.detach()):.6f}"
    elif algorithm == "DataDAM":
        synthetic = make_synthetic(images)
        loss = run_datadam_matching(module, model, images, synthetic, num_classes)
        require_input_gradient(synthetic)
        torch.optim.SGD([synthetic], lr=0.01).step()
        detail = f"attention_match_loss={float(loss.detach()):.6f}"
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
