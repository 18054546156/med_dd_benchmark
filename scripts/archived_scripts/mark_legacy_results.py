#!/usr/bin/env python3
"""标记 Legacy 结果

将已有的实验结果标记为 legacy_native 协议，不删除但不作为公平比较结果。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def create_legacy_manifest(
    result_dir: Path,
    algorithm: str,
    dataset: str,
    description: str
) -> Path:
    """
    为 legacy 结果创建 manifest

    Args:
        result_dir: 结果目录
        algorithm: 算法名称
        dataset: 数据集名称
        description: 描述信息

    Returns:
        manifest 文件路径
    """
    manifest = {
        "protocol": "legacy_native",
        "algorithm": algorithm,
        "dataset": dataset,
        "description": description,
        "warning": "These results used algorithm-native configurations with inconsistent backbones. "
                  "Not suitable for direct comparison across algorithms.",
        "backbone_note": "Backbone architecture varies by algorithm. See individual config files.",
        "usage": "Reference only. Do not use for fair comparison tables.",
        "marked_at": "2026-08-12"
    }

    manifest_path = result_dir / "LEGACY_MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


def create_legacy_readme(result_dir: Path, algorithm: str, dataset: str) -> Path:
    """
    为 legacy 结果创建 README

    Returns:
        README 文件路径
    """
    readme_content = f"""# Legacy Results - {algorithm} / {dataset}

## ⚠️ WARNING

These results are from the **legacy_native** protocol and use algorithm-specific configurations with **inconsistent backbones**.

**DO NOT use these results for direct comparison across algorithms.**

## Why Legacy?

- **Inconsistent backbones**: Different algorithms used different model depths/normalizations
  - Example: COVID dataset had MTT using D4, HoP-TM using D5, NCFM using D4, DataDAM using D3+BN
- **No unified evaluation protocol**: Different evaluation settings across runs
- **Incomplete reproducibility records**: Missing seeds, configs, or environment details

## What Should You Use?

Use results from the **fair_comparison_v1** protocol instead:
- Unified backbone for each dataset
- Consistent evaluation protocol
- Complete reproducibility manifests

See: `configs/backbones/README.md` for fair comparison guidelines

## Can I Still Use These?

✅ **YES** for:
- Understanding algorithm behavior in its native setting
- Debugging or development reference
- Approximate ballpark numbers

❌ **NO** for:
- Direct algorithm-to-algorithm comparison
- Published tables or figures
- Claims about relative algorithm performance

## Migration

To run fair comparison experiments:

```bash
# Use fair configs
python scripts/run_config.py \\
  --config configs/<algorithm>/<dataset>/fair_ipc10.yaml \\
  --algorithm <algorithm> \\
  --stage distill \\
  --run

# Results will go to:
# results/<algorithm>/<dataset>/<backbone>/seed<N>/<timestamp>/
```

---

**Marked as legacy on: 2026-08-12**
"""

    readme_path = result_dir / "LEGACY_README.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme_content)

    return readme_path


def mark_legacy_directory(
    target_dir: Path,
    algorithm: str,
    dataset: str,
    dry_run: bool = False
) -> Dict[str, Path]:
    """
    标记单个 legacy 目录

    Returns:
        创建的文件路径字典
    """
    if not target_dir.exists():
        print(f"  WARNING: Directory does not exist: {target_dir}")
        return {}

    print(f"  Marking: {target_dir}")

    created = {}

    if not dry_run:
        # 创建 manifest
        manifest_path = create_legacy_manifest(
            target_dir,
            algorithm,
            dataset,
            f"Legacy {algorithm} results on {dataset} with native configuration"
        )
        created["manifest"] = manifest_path

        # 创建 README
        readme_path = create_legacy_readme(target_dir, algorithm, dataset)
        created["readme"] = readme_path

        print(f"    Created: {manifest_path.name}")
        print(f"    Created: {readme_path.name}")
    else:
        print(f"    [DRY RUN] Would create LEGACY_MANIFEST.json and LEGACY_README.md")

    return created


def scan_legacy_results(root: Path) -> List[Dict[str, any]]:
    """
    扫描需要标记的 legacy 目录

    Returns:
        待标记目录列表
    """
    targets = []

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

                # 检查是否已经标记
                if (dataset_dir / "LEGACY_MANIFEST.json").exists():
                    continue

                targets.append({
                    "type": "results",
                    "algorithm": algorithm,
                    "dataset": dataset,
                    "path": dataset_dir
                })

    # 扫描 pretrained_models/
    pretrain_root = root / "pretrained_models"
    if pretrain_root.exists():
        for dataset_dir in pretrain_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            if (dataset_dir / "LEGACY_MANIFEST.json").exists():
                continue

            targets.append({
                "type": "pretrained_models",
                "algorithm": "ncfm",  # 假设是 NCFM
                "dataset": dataset,
                "path": dataset_dir
            })

    return targets


def main():
    parser = argparse.ArgumentParser(description="标记 legacy 结果")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="项目根目录"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印不实际创建文件"
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="手动指定要标记的目录"
    )
    parser.add_argument(
        "--algorithm",
        help="手动指定算法名称（与 --target 一起使用）"
    )
    parser.add_argument(
        "--dataset",
        help="手动指定数据集名称（与 --target 一起使用）"
    )

    args = parser.parse_args()

    if args.target:
        # 手动标记单个目录
        if not args.algorithm or not args.dataset:
            print("ERROR: --algorithm and --dataset required when using --target")
            return 1

        mark_legacy_directory(args.target, args.algorithm, args.dataset, args.dry_run)
    else:
        # 自动扫描并标记
        print(f"Scanning for legacy results in: {args.root}")
        targets = scan_legacy_results(args.root)

        print(f"\nFound {len(targets)} directories to mark as legacy")

        if not targets:
            print("✅ No unmarked legacy directories found")
            return 0

        for target in targets:
            print(f"\n{target['type']}: {target['algorithm']}/{target['dataset']}")
            mark_legacy_directory(
                target["path"],
                target["algorithm"],
                target["dataset"],
                args.dry_run
            )

        if args.dry_run:
            print("\n[DRY RUN] No files were actually created. Run without --dry-run to apply.")
        else:
            print(f"\n✅ Marked {len(targets)} directories as legacy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
