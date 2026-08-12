#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据已经落盘的 train/val/test 目录刷新数据集 manifest。

这个脚本只统计现有文件，不重新切分、不复制和不修改原始数据。PathMNIST
转换完成后应运行一次，确保 manifest 与实际 ImageFolder 文件数一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def refresh(dataset_root: Path) -> Path:
    """统计三个 split 的类别文件，并原子更新 manifest.json。"""
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少数据集 manifest: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    files: list[dict[str, object]] = []

    for split in ("train", "val", "test"):
        split_root = dataset_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"缺少数据划分目录: {split_root}")
        split_total = 0
        for class_root in sorted(item for item in split_root.iterdir() if item.is_dir()):
            image_files = sorted(item for item in class_root.iterdir() if item.is_file())
            count = len(image_files)
            counts[f"{split}/{class_root.name}"] = count
            split_total += count
            # 只记录相对路径和类别，不重复写入大尺寸图像元数据。
            files.extend(
                {
                    "target": str(path.relative_to(dataset_root)).replace("/", "\\"),
                    "split": split,
                    "class_name": class_root.name,
                    "class_index": int(class_root.name) if class_root.name.isdigit() else None,
                }
                for path in image_files
            )
        split_counts[split] = split_total

    payload["format"] = "ImageFolder (train/val/test/class/image)"
    payload["splits"] = split_counts
    payload["counts"] = counts
    payload["num_files"] = sum(split_counts.values())
    payload["files"] = files

    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新 prepared 数据集 manifest")
    parser.add_argument("dataset_root", type=Path, help="例如 data/prepared/PathMNIST")
    args = parser.parse_args()
    path = refresh(args.dataset_root.resolve())
    print(f"Manifest refreshed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
