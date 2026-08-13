#!/usr/bin/env python3
"""配置文件校验工具

检查配置文件的问题：
1. 同名键冲突（flatten_dict 会覆盖）
2. 必需字段缺失
3. 类型错误
4. Backbone 不一致（fair comparison）
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


def flatten_dict_with_tracking(d: Dict[str, Any], parent_key: str = '') -> Dict[str, List[str]]:
    """
    扁平化字典并追踪每个键的来源路径

    Returns:
        {final_key: [source_path1, source_path2, ...]}
    """
    items = {}

    for k, v in d.items():
        # 当前路径
        current_path = f"{parent_key}.{k}" if parent_key else k

        # 最终的扁平键（NCFM ArgsProcessor 的逻辑）
        flat_key = k

        if isinstance(v, dict):
            nested = flatten_dict_with_tracking(v, current_path)
            for nk, paths in nested.items():
                if nk not in items:
                    items[nk] = []
                items[nk].extend(paths)
        else:
            if flat_key not in items:
                items[flat_key] = []
            items[flat_key].append(current_path)

    return items


def check_key_conflicts(config_path: Path, algorithm: str) -> List[Dict[str, Any]]:
    """
    检查同名键冲突

    Returns:
        问题列表
    """
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    flat = flatten_dict_with_tracking(config)

    issues = []
    # 这些键在 MTT 中分别属于 buffer 和 distillation 两个阶段，
    # 分阶段重复是必要的，不会被同一个入口同时展开覆盖。
    intentional_mtt = {"model", "buffer_path", "batch_train", "batch_real", "lr_teacher"}
    metadata_keys = {"note"}
    for key, paths in flat.items():
        if key in metadata_keys:
            continue
        if algorithm == "mtt" and key in intentional_mtt:
            continue
        if len(paths) > 1:
            issues.append({
                "type": "key_conflict",
                "severity": "error",
                "key": key,
                "paths": paths,
                "message": f"Key '{key}' defined in multiple places: {', '.join(paths)}. "
                          f"flatten_dict() will use the last one unpredictably."
            })

    return issues


def check_required_fields(config_path: Path, algorithm: str, dataset: str) -> List[Dict[str, Any]]:
    """
    检查必需字段

    Returns:
        问题列表
    """
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required = {
        "all": ["dataset", "num_classes", "channel", "im_size", "ipc"],
        "ncfm": ["nclass", "nch", "size", "depth", "norm_type", "lr", "pertrain_epochs", "niter"],
        "cafe": ["lr_img", "lr_net", "Iteration", "batch_real"],
        "datadam": ["lr_img", "lr_net", "Iteration", "task_balance"],
        "hop_tm": ["model", "lr_img", "lr_teacher", "Iteration"],
        "mtt": ["model", "lr_img", "lr_teacher", "Iteration"],
    }

    issues = []
    # NCFM 使用 dataset/network/train/condense 的原始嵌套合同，
    # 不应被要求提供其它算法的顶层 num_classes/channel/im_size。
    required_keys = required.get(algorithm, []) if algorithm == "ncfm" else required.get("all", []) + required.get(algorithm, [])

    def get_nested(obj, key):
        """递归查找嵌套键"""
        if key in obj:
            return obj[key]
        for v in obj.values():
            if isinstance(v, dict):
                result = get_nested(v, key)
                if result is not None:
                    return result
        return None

    for key in required_keys:
        value = get_nested(config, key)
        if value is None:
            issues.append({
                "type": "missing_required",
                "severity": "error",
                "key": key,
                "message": f"Required field '{key}' not found for {algorithm}/{dataset}"
            })

    return issues


def check_backbone_consistency(config_path: Path, backbone_path: Path) -> List[Dict[str, Any]]:
    """
    检查配置是否与 backbone 一致

    Returns:
        问题列表
    """
    if not backbone_path.exists():
        return [{
            "type": "missing_backbone",
            "severity": "warning",
            "message": f"Backbone config not found: {backbone_path}"
        }]

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    with backbone_path.open("r", encoding="utf-8") as f:
        backbone = yaml.safe_load(f) or {}

    issues = []

    # 需要一致的字段
    consistency_keys = ["num_classes", "channel", "im_size", "depth", "width", "norm", "model"]

    def get_value(obj, key):
        if key in obj:
            return obj[key]
        for section in ["network", "model_config", "backbone"]:
            if section in obj and isinstance(obj[section], dict) and key in obj[section]:
                return obj[section][key]
        return None

    for key in consistency_keys:
        config_val = get_value(config, key)
        backbone_val = get_value(backbone, key)

        # NCFM 的 width 是相对基宽的倍率（1.0 * 128），而统一合同
        # 文件记录的是实际通道宽度 128；两者语义不同，不能直接比较。
        if key == "width" and config.get("network", {}).get("net_type") == "convnet":
            config_val = int(128 * float(config_val))

        if backbone_val is not None and config_val is not None:
            if config_val != backbone_val:
                issues.append({
                    "type": "backbone_mismatch",
                    "severity": "error",
                    "key": key,
                    "config_value": config_val,
                    "backbone_value": backbone_val,
                    "message": f"Field '{key}' mismatch: config={config_val}, backbone={backbone_val}"
                })

    return issues


def validate_config(
    config_path: Path,
    algorithm: str,
    dataset: str,
    backbone_dir: Path
) -> Dict[str, Any]:
    """
    完整校验单个配置文件

    Returns:
        校验报告
    """
    report = {
        "config": str(config_path),
        "algorithm": algorithm,
        "dataset": dataset,
        "issues": []
    }

    # 1. 检查键冲突
    report["issues"].extend(check_key_conflicts(config_path, algorithm))

    # 2. 检查必需字段
    report["issues"].extend(check_required_fields(config_path, algorithm, dataset))

    # 3. 检查 backbone 一致性
    backbone_path = backbone_dir / f"fair_{dataset.lower()}.yaml"
    report["issues"].extend(check_backbone_consistency(config_path, backbone_path))

    return report


def main():
    parser = argparse.ArgumentParser(description="校验配置文件")
    parser.add_argument(
        "--config",
        type=Path,
        help="配置文件路径（不指定则校验所有）"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="项目根目录"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式（warning 也算失败）"
    )

    args = parser.parse_args()

    configs_dir = args.root / "configs"
    backbone_dir = configs_dir / "backbones"

    if args.config:
        # 校验单个文件
        # 从路径推断 algorithm 和 dataset
        parts = args.config.parts
        try:
            algo_idx = parts.index("configs") + 1
            algorithm = parts[algo_idx]
            dataset = parts[algo_idx + 1].upper()
        except (ValueError, IndexError):
            print(f"ERROR: Cannot infer algorithm/dataset from path: {args.config}")
            return 1

        reports = [validate_config(args.config, algorithm, dataset, backbone_dir)]
    else:
        # 校验所有配置
        reports = []
        for algo_dir in configs_dir.iterdir():
            if not algo_dir.is_dir() or algo_dir.name == "backbones":
                continue
            algorithm = algo_dir.name

            for dataset_dir in algo_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                dataset = dataset_dir.name.upper()

                for config_file in dataset_dir.glob("*.yaml"):
                    if "smoke" in config_file.name:
                        continue  # 跳过 smoke test 配置
                    reports.append(validate_config(config_file, algorithm, dataset, backbone_dir))

    # 打印报告
    total_errors = 0
    total_warnings = 0

    for report in reports:
        if not report["issues"]:
            continue

        print(f"\n{'='*60}")
        print(f"Config: {report['config']}")
        print(f"Algorithm: {report['algorithm']}, Dataset: {report['dataset']}")
        print(f"{'='*60}")

        for issue in report["issues"]:
            if issue["severity"] == "error":
                total_errors += 1
                symbol = "❌"
            elif issue["severity"] == "warning":
                total_warnings += 1
                symbol = "⚠️"
            else:
                symbol = "ℹ️"

            print(f"\n{symbol} {issue['type'].upper()} ({issue['severity']})")
            print(f"   {issue['message']}")
            if "paths" in issue:
                for path in issue["paths"]:
                    print(f"     - {path}")

    # 汇总
    print(f"\n\n{'='*60}")
    print(f"SUMMARY: Validated {len(reports)} configs")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")
    print(f"{'='*60}")

    if total_errors > 0:
        return 1
    if args.strict and total_warnings > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
