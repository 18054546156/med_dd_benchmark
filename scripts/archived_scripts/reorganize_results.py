#!/usr/bin/env python3
"""重组实验输出目录为分层结构

旧结构（扁平，难以管理）：
    results/ncfm/covid/condense/covid/ipc10/data_20000.pt
    pretrained_models/covid/model_19.pth
    logs/ncfm-pipeline/covid-24978.out

新结构（分层，易于追溯）：
    results/
      <algorithm>/
        <dataset>/
          <backbone>/
            <seed>/
              <run_id>/
                config.yaml
                manifest.json
                checkpoints/
                synthetic_data/
                logs/
                  stdout.log
                  stderr.log

示例：
    results/ncfm/COVID/ConvNetD4_IN_W128/seed0/20260812_143022/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional


def parse_backbone_name(
    model: str,
    depth: Optional[int] = None,
    width: Optional[int] = None,
    norm: Optional[str] = None
) -> str:
    """
    生成标准化的 backbone 名称

    Example: ConvNet + D4 + IN + W128 -> ConvNetD4_IN_W128
    """
    parts = [model]
    if depth is not None:
        parts.append(f"D{depth}")
    if norm:
        norm_abbr = {
            "instancenorm": "IN",
            "batchnorm": "BN",
            "groupnorm": "GN",
            "none": "NoNorm"
        }.get(norm.lower(), norm.upper())
        parts.append(norm_abbr)
    if width is not None:
        parts.append(f"W{width}")
    return "_".join(parts)


def create_hierarchical_structure(
    root: Path,
    algorithm: str,
    dataset: str,
    backbone: str,
    seed: int,
    run_id: str
) -> Path:
    """
    创建分层目录结构

    Returns:
        实验输出根目录
    """
    exp_dir = root / algorithm / dataset / backbone / f"seed{seed}" / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "synthetic_data").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    return exp_dir


def migrate_experiment(
    old_results_dir: Path,
    old_pretrain_dir: Optional[Path],
    old_logs_dir: Optional[Path],
    new_root: Path,
    algorithm: str,
    dataset: str,
    backbone_config: dict,
    seed: int,
    run_id: str,
    dry_run: bool = False
) -> Path:
    """
    迁移单个实验的所有产物到新目录结构

    Args:
        old_results_dir: 旧的 results 目录
        old_pretrain_dir: 旧的 pretrained_models 目录
        old_logs_dir: 旧的 logs 目录
        new_root: 新的根目录
        algorithm: 算法名称
        dataset: 数据集名称
        backbone_config: backbone 配置字典
        seed: 随机种子
        run_id: 运行 ID (时间戳)
        dry_run: 是否只打印不实际移动

    Returns:
        新的实验目录路径
    """
    backbone_name = parse_backbone_name(
        model=backbone_config.get("model", "ConvNet"),
        depth=backbone_config.get("depth"),
        width=backbone_config.get("width", 128),
        norm=backbone_config.get("norm", "instancenorm")
    )

    exp_dir = create_hierarchical_structure(
        new_root, algorithm, dataset, backbone_name, seed, run_id
    )

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating to: {exp_dir}")

    # 迁移 checkpoints
    if old_pretrain_dir and old_pretrain_dir.exists():
        ckpt_files = list(old_pretrain_dir.glob("*.pth")) + list(old_pretrain_dir.glob("*.pth.tar"))
        print(f"  Checkpoints: {len(ckpt_files)} files")
        for ckpt in ckpt_files:
            target = exp_dir / "checkpoints" / ckpt.name
            if not dry_run:
                shutil.copy2(ckpt, target)

    # 迁移 synthetic data
    if old_results_dir.exists():
        syn_files = list(old_results_dir.glob("*.pt")) + list(old_results_dir.glob("*.npz"))
        print(f"  Synthetic data: {len(syn_files)} files")
        for syn in syn_files:
            target = exp_dir / "synthetic_data" / syn.name
            if not dry_run:
                shutil.copy2(syn, target)

    # 迁移 logs
    if old_logs_dir and old_logs_dir.exists():
        log_files = list(old_logs_dir.glob("*.log")) + list(old_logs_dir.glob("*.out")) + list(old_logs_dir.glob("*.err"))
        print(f"  Logs: {len(log_files)} files")
        for log in log_files:
            target = exp_dir / "logs" / log.name
            if not dry_run:
                shutil.copy2(log, target)

    return exp_dir


def scan_legacy_results(root: Path) -> list:
    """
    扫描旧的扁平结构，识别需要迁移的实验

    Returns:
        实验信息列表
    """
    experiments = []

    # 扫描 results/
    results_root = root / "results"
    if results_root.exists():
        for algo_dir in results_root.iterdir():
            if not algo_dir.is_dir():
                continue
            algorithm = algo_dir.name

            for dataset_dir in algo_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                dataset = dataset_dir.name

                # 查找所有可能的结果目录
                for subdir in dataset_dir.rglob("*"):
                    if subdir.is_dir() and (
                        list(subdir.glob("*.pt")) or
                        list(subdir.glob("*.pth"))
                    ):
                        experiments.append({
                            "algorithm": algorithm,
                            "dataset": dataset,
                            "results_dir": subdir,
                            "type": "results"
                        })

    return experiments


def main():
    parser = argparse.ArgumentParser(description="重组实验目录为分层结构")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="项目根目录"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="新的输出根目录（默认为 root/results_fair）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印不实际移动文件"
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="只扫描旧结构，不迁移"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.root / "results_fair"

    print(f"Scanning legacy results in: {args.root}")
    experiments = scan_legacy_results(args.root)

    print(f"\nFound {len(experiments)} experiment directories")

    if args.scan_only:
        for exp in experiments:
            print(f"  {exp['algorithm']}/{exp['dataset']}: {exp['results_dir']}")
        return 0

    # TODO: 实际迁移需要读取每个实验的配置文件来获取 backbone 信息
    print("\nWARNING: Automatic migration requires experiment manifest or config files")
    print("Use --scan-only to see what would be migrated")
    print("\nFor manual migration, use:")
    print("  python scripts/create_manifest.py --config <config> --algorithm <algo> ...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
