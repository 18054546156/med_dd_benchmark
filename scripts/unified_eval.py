#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一评估脚本：公平比较不同DD算法生成的合成数据集。

本脚本读取任意DD算法生成的合成数据（.pt文件），使用完全相同的评估协议：
- 固定训练轮数：epoch_eval_train=1000
- 固定优化器：SGD(lr=0.01, momentum=0.9, weight_decay=0.0005)
- 固定学习率调度：[epoch//2+1] 处降为0.1倍
- 固定评估次数：num_eval=5（重复5次取平均）
- 固定随机种子策略
- 统一的测试集：data/prepared/<dataset>/test/

这确保不同算法的accuracy结果可以公平对比。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.medical_dataset_utils import load_medical_splits, MEDICAL_DATASET_SPECS


# ============================================================================
# 统一评估配置（所有算法使用相同的参数）
# ============================================================================

UNIFIED_EVAL_CONFIG = {
    "epoch_eval_train": 1000,      # 训练轮数
    "lr_net": 0.01,                # 初始学习率
    "momentum": 0.9,               # SGD动量
    "weight_decay": 0.0005,        # 权重衰减
    "batch_train": 256,            # 训练batch size
    "num_eval": 5,                 # 评估重复次数
    "lr_schedule_ratio": 0.5,      # 学习率调度：在epoch*ratio处降低
    "lr_decay": 0.1,               # 学习率衰减倍数
}


# ============================================================================
# ConvNet 网络定义（复用自adapted/dc_dsa_dm/networks.py）
# ============================================================================

class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=(32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


def get_network(model: str, channel: int, num_classes: int, im_size: tuple[int, int]) -> nn.Module:
    """创建ConvNet网络。

    Args:
        model: 模型名称（ConvNet/ConvNetD3/ConvNetD5等）
        channel: 输入通道数
        num_classes: 类别数
        im_size: 图像尺寸

    Returns:
        网络实例
    """
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'ConvNet':
        net_depth = 3
    elif model == 'ConvNetD1':
        net_depth = 1
    elif model == 'ConvNetD2':
        net_depth = 2
    elif model == 'ConvNetD3':
        net_depth = 3
    elif model == 'ConvNetD4':
        net_depth = 4
    elif model == 'ConvNetD5':
        net_depth = 5
    else:
        raise ValueError(f"未知模型: {model}")

    net = ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size)
    return net


def get_default_convnet_setting():
    """获取ConvNet默认配置。"""
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


# ============================================================================
# 训练和评估函数
# ============================================================================

def set_seed(seed: int):
    """设置所有随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(net, dataloader, optimizer, criterion, device):
    """训练一个epoch。"""
    net.train()
    loss_avg, acc_avg, num_exp = 0, 0, 0

    for i_batch, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg += loss.item() * images.shape[0]
        acc = (output.argmax(dim=1) == labels).float().sum()
        acc_avg += acc.item()
        num_exp += images.shape[0]

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def test_epoch(net, dataloader, criterion, device):
    """测试一个epoch。"""
    net.eval()
    loss_avg, acc_avg, num_exp = 0, 0, 0

    with torch.no_grad():
        for i_batch, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            output = net(images)
            loss = criterion(output, labels)

            loss_avg += loss.item() * images.shape[0]
            acc = (output.argmax(dim=1) == labels).float().sum()
            acc_avg += acc.item()
            num_exp += images.shape[0]

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset_unified(
    images_train: torch.Tensor,
    labels_train: torch.Tensor,
    test_dataset,
    model: str,
    channel: int,
    num_classes: int,
    im_size: tuple[int, int],
    device: str,
    seed: int,
) -> dict[str, Any]:
    """使用统一协议评估合成数据集。

    Args:
        images_train: 合成图像 [N, C, H, W]
        labels_train: 合成标签 [N]
        test_dataset: 测试集
        model: 模型名称
        channel: 输入通道数
        num_classes: 类别数
        im_size: 图像尺寸
        device: 设备
        seed: 随机种子

    Returns:
        包含训练loss/acc和测试loss/acc的字典
    """
    set_seed(seed)

    # 创建网络
    net = get_network(model, channel, num_classes, im_size)
    net = net.to(device)

    # 准备数据
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)

    # 创建训练集
    dst_train = TensorDataset(images_train, labels_train)
    trainloader = DataLoader(dst_train, batch_size=UNIFIED_EVAL_CONFIG["batch_train"], shuffle=True, num_workers=0)

    # 创建测试集
    testloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # 优化器和损失函数
    lr = UNIFIED_EVAL_CONFIG["lr_net"]
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=UNIFIED_EVAL_CONFIG["momentum"],
        weight_decay=UNIFIED_EVAL_CONFIG["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss().to(device)

    # 学习率调度
    Epoch = UNIFIED_EVAL_CONFIG["epoch_eval_train"]
    lr_schedule_epoch = int(Epoch * UNIFIED_EVAL_CONFIG["lr_schedule_ratio"]) + 1

    # 训练
    start_time = time.time()
    for ep in range(Epoch + 1):
        loss_train, acc_train = train_epoch(net, trainloader, optimizer, criterion, device)

        # 学习率调度
        if ep == lr_schedule_epoch:
            lr *= UNIFIED_EVAL_CONFIG["lr_decay"]
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=lr,
                momentum=UNIFIED_EVAL_CONFIG["momentum"],
                weight_decay=UNIFIED_EVAL_CONFIG["weight_decay"]
            )

    train_time = time.time() - start_time

    # 测试
    loss_test, acc_test = test_epoch(net, testloader, criterion, device)

    return {
        "train_loss": loss_train,
        "train_acc": acc_train * 100,  # 转换为百分比
        "test_loss": loss_test,
        "test_acc": acc_test * 100,    # 转换为百分比
        "train_time": train_time,
        "seed": seed,
    }


# ============================================================================
# 数据加载函数
# ============================================================================

def load_synthetic_data(data_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """加载合成数据。

    支持的格式:
    1. 单个.pt文件（DC/DSA/DM/DataDAM/CAFE格式）
    2. MTT/HoP-TM格式（images_best.pt + labels_best.pt）
    3. NCFM格式（data_*.pt）

    Args:
        data_path: 数据文件或目录路径

    Returns:
        (images, labels) 张量
    """
    data_path = Path(data_path)

    if data_path.is_file():
        # 单个文件
        if data_path.suffix != '.pt':
            raise ValueError(f"数据文件必须是.pt格式: {data_path}")

        data = torch.load(data_path, map_location='cpu')

        # 检查数据格式
        if isinstance(data, dict):
            # NCFM格式: {'data': [...], 'label': [...]}
            if 'data' in data and 'label' in data:
                images = torch.stack(data['data'])
                labels = torch.tensor(data['label'])
            # 其他字典格式
            elif 'images' in data and 'labels' in data:
                images = data['images']
                labels = data['labels']
            else:
                raise ValueError(f"未知的数据格式（字典）: {list(data.keys())}")
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            # 元组格式: (images, labels)
            images, labels = data
            if not isinstance(images, torch.Tensor):
                images = torch.stack(images)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
        else:
            raise ValueError(f"未知的数据格式: {type(data)}")

    elif data_path.is_dir():
        # 目录：查找images_best.pt和labels_best.pt（MTT/HoP-TM格式）
        images_file = data_path / "images_best.pt"
        labels_file = data_path / "labels_best.pt"

        if images_file.exists() and labels_file.exists():
            images = torch.load(images_file, map_location='cpu')
            labels = torch.load(labels_file, map_location='cpu')
        else:
            raise FileNotFoundError(
                f"目录 {data_path} 中未找到 images_best.pt 和 labels_best.pt"
            )
    else:
        raise FileNotFoundError(f"数据路径不存在: {data_path}")

    # 确保标签是1D张量
    if labels.dim() > 1:
        labels = labels.squeeze()

    return images, labels


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="统一评估脚本：公平比较不同DD算法生成的合成数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估DC算法生成的数据
  python scripts/unified_eval.py \\
    --data results/dc_dsa_dm/COVID/DC/ipc10/res_DC_COVID_ConvNetD5_10ipc.pt \\
    --dataset COVID \\
    --model ConvNetD5

  # 评估MTT算法生成的数据（目录格式）
  python scripts/unified_eval.py \\
    --data results/mtt/COVID/ipc10/COVID/run_001/ \\
    --dataset COVID \\
    --model ConvNetD5

  # 评估NCFM算法生成的数据
  python scripts/unified_eval.py \\
    --data results/ncfm/covid/condense/covid/ipc10/run_001/distilled_data/data_20000.pt \\
    --dataset COVID \\
    --model ConvNetD5
        """
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="合成数据路径（.pt文件或包含images_best.pt的目录）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PathMNIST", "COVID", "Kvasir"],
        help="数据集名称"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ConvNet", "ConvNetD1", "ConvNetD2", "ConvNetD3", "ConvNetD4", "ConvNetD5"],
        help="评估网络模型"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="设备选择（auto自动检测）"
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=None,
        help=f"评估重复次数（默认{UNIFIED_EVAL_CONFIG['num_eval']}）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="结果输出路径（JSON格式，默认打印到终端）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="基础随机种子（每次评估使用seed+i）"
    )

    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"使用设备: {device}")

    # 评估次数
    num_eval = args.num_eval if args.num_eval is not None else UNIFIED_EVAL_CONFIG["num_eval"]

    # 加载合成数据
    print(f"\n加载合成数据: {args.data}")
    try:
        images_syn, labels_syn = load_synthetic_data(args.data)
    except Exception as e:
        print(f"错误: 加载数据失败 - {e}")
        return 1

    print(f"合成数据形状: images={images_syn.shape}, labels={labels_syn.shape}")

    # 获取数据集规格
    if args.dataset not in MEDICAL_DATASET_SPECS:
        print(f"错误: 未知数据集 {args.dataset}")
        return 1

    spec = MEDICAL_DATASET_SPECS[args.dataset]
    channel = spec["channel"]
    num_classes = spec["num_classes"]
    im_size = tuple(spec["im_size"])

    # 加载测试集
    print(f"\n加载测试集: data/prepared/{args.dataset}/test/")
    data_root = PROJECT_ROOT / "data" / "prepared" / args.dataset
    try:
        _, _, test_dataset = load_medical_splits(data_root)
    except Exception as e:
        print(f"错误: 加载测试集失败 - {e}")
        return 1

    print(f"测试集大小: {len(test_dataset)}")

    # 统一评估配置
    print(f"\n统一评估配置:")
    print(f"  - 训练轮数: {UNIFIED_EVAL_CONFIG['epoch_eval_train']}")
    print(f"  - 学习率: {UNIFIED_EVAL_CONFIG['lr_net']}")
    print(f"  - 动量: {UNIFIED_EVAL_CONFIG['momentum']}")
    print(f"  - 权重衰减: {UNIFIED_EVAL_CONFIG['weight_decay']}")
    print(f"  - Batch大小: {UNIFIED_EVAL_CONFIG['batch_train']}")
    print(f"  - 评估次数: {num_eval}")
    print(f"  - 模型: {args.model}")

    # 多次评估
    print(f"\n开始评估...")
    results = []

    for i in range(num_eval):
        seed = args.seed + i
        print(f"\n[{i+1}/{num_eval}] 评估 (seed={seed})")

        result = evaluate_synset_unified(
            images_syn,
            labels_syn,
            test_dataset,
            args.model,
            channel,
            num_classes,
            im_size,
            device,
            seed,
        )

        results.append(result)

        print(f"  训练: loss={result['train_loss']:.4f}, acc={result['train_acc']:.2f}%")
        print(f"  测试: loss={result['test_loss']:.4f}, acc={result['test_acc']:.2f}%")
        print(f"  时间: {result['train_time']:.1f}s")

    # 计算统计结果
    test_accs = [r['test_acc'] for r in results]
    train_accs = [r['train_acc'] for r in results]

    summary = {
        "data_path": str(args.data),
        "dataset": args.dataset,
        "model": args.model,
        "num_synthetic_images": int(images_syn.shape[0]),
        "num_eval": num_eval,
        "eval_config": UNIFIED_EVAL_CONFIG,
        "test_accuracy": {
            "mean": float(np.mean(test_accs)),
            "std": float(np.std(test_accs)),
            "min": float(np.min(test_accs)),
            "max": float(np.max(test_accs)),
            "all": test_accs,
        },
        "train_accuracy": {
            "mean": float(np.mean(train_accs)),
            "std": float(np.std(train_accs)),
        },
        "individual_results": results,
    }

    # 打印总结
    print(f"\n{'='*60}")
    print(f"统一评估结果总结")
    print(f"{'='*60}")
    print(f"数据路径: {args.data}")
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"合成图像数量: {images_syn.shape[0]}")
    print(f"评估次数: {num_eval}")
    print(f"\n测试准确率:")
    print(f"  平均: {summary['test_accuracy']['mean']:.2f}%")
    print(f"  标准差: {summary['test_accuracy']['std']:.2f}%")
    print(f"  范围: [{summary['test_accuracy']['min']:.2f}%, {summary['test_accuracy']['max']:.2f}%]")
    print(f"\n训练准确率:")
    print(f"  平均: {summary['train_accuracy']['mean']:.2f}%")
    print(f"  标准差: {summary['train_accuracy']['std']:.2f}%")
    print(f"{'='*60}")

    # 保存结果
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
