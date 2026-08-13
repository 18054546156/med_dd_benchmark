#!/usr/bin/env python3
"""将 PathMNIST NPZ 转换为 train/val/test 目录结构，与 COVID/Kvasir 保持一致。

使用官方 MedMNIST 的 train/val/test 划分，不重新随机切分。
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_pathmnist(data_root: Path, force: bool = False):
    """
    将 pathmnist.npz 转换为目录结构。

    Args:
        data_root: data/prepared 根目录
        force: 是否覆盖已有目录
    """
    pathmnist_dir = data_root / "PathMNIST"
    npz_path = pathmnist_dir / "pathmnist.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"PathMNIST NPZ 不存在: {npz_path}")

    # 检查是否已经转换
    train_dir = pathmnist_dir / "train"
    val_dir = pathmnist_dir / "val"
    test_dir = pathmnist_dir / "test"

    if train_dir.exists() or val_dir.exists() or test_dir.exists():
        if not force:
            print(f"WARNING: PathMNIST directories already exist, use --force to overwrite")
            print(f"   train: {train_dir.exists()}")
            print(f"   val:   {val_dir.exists()}")
            print(f"   test:  {test_dir.exists()}")
            return
        else:
            print("Removing existing directories...")
            for d in [train_dir, val_dir, test_dir]:
                if d.exists():
                    shutil.rmtree(d)

    # 加载 NPZ
    print(f"Loading {npz_path.name}...")
    data = np.load(npz_path)

    # 检查 NPZ 包含的 keys
    print(f"   Keys: {list(data.keys())}")

    # 官方 MedMNIST NPZ 格式
    splits = {
        "train": {
            "images": data["train_images"],
            "labels": data["train_labels"].squeeze()
        },
        "val": {
            "images": data["val_images"],
            "labels": data["val_labels"].squeeze()
        },
        "test": {
            "images": data["test_images"],
            "labels": data["test_labels"].squeeze()
        }
    }

    # 统计信息
    stats = {}
    for split_name, split_data in splits.items():
        images = split_data["images"]
        labels = split_data["labels"]
        unique_labels = np.unique(labels)

        print(f"\n{split_name}:")
        print(f"   Images: {len(images)}")
        print(f"   Shape:  {images.shape}")
        print(f"   Labels: {len(unique_labels)} classes")
        print(f"   Range:  [{labels.min()}, {labels.max()}]")

        stats[split_name] = {
            "num_images": int(len(images)),
            "shape": list(images.shape),
            "num_classes": int(len(unique_labels)),
            "class_counts": {int(c): int(np.sum(labels == c)) for c in unique_labels}
        }

    # 转换为目录结构
    print(f"\nConverting to directory structure...")

    num_classes = 9  # PathMNIST 有 9 个类别

    for split_name, split_data in splits.items():
        split_dir = pathmnist_dir / split_name
        split_dir.mkdir(exist_ok=True)

        images = split_data["images"]
        labels = split_data["labels"]

        # 创建类别目录
        for c in range(num_classes):
            (split_dir / str(c)).mkdir(exist_ok=True)

        # 保存图像
        print(f"   {split_name}: ", end="", flush=True)
        for idx, (img_array, label) in enumerate(tqdm(zip(images, labels),
                                                       total=len(images),
                                                       desc=split_name,
                                                       leave=False)):
            # PathMNIST 图像是 28x28 RGB
            # 保存为原始 28x28，resize 由 loader 处理
            if img_array.ndim == 2:
                # 如果是灰度图，转为 RGB
                img_array = np.stack([img_array] * 3, axis=-1)

            img = Image.fromarray(img_array.astype(np.uint8))
            img_path = split_dir / str(label) / f"{idx:05d}.png"
            img.save(img_path)

        print(f"DONE: {len(images)} images")

    # 更新 manifest
    manifest_path = pathmnist_dir / "manifest.json"
    manifest = json.load(manifest_path.open("r"))

    manifest["format"] = "ImageFolder (train/val/test/class/image.png)"
    manifest["splits"] = {
        "train": stats["train"]["num_images"],
        "val": stats["val"]["num_images"],
        "test": stats["test"]["num_images"]
    }
    manifest["split_source"] = "official_medmnist_v2"
    manifest["num_classes"] = num_classes
    manifest["image_size"] = [28, 28]  # 原始尺寸
    manifest["note"] = "Official MedMNIST train/val/test split, not re-sampled. Loader resizes to 32x32."
    manifest["class_distribution"] = stats

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nConversion complete:")
    print(f"   train: {stats['train']['num_images']} images")
    print(f"   val:   {stats['val']['num_images']} images")
    print(f"   test:  {stats['test']['num_images']} images")
    print(f"   manifest: {manifest_path}")

    # 提示是否保留 NPZ
    npz_size_mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"\nOriginal NPZ: {npz_path.name} ({npz_size_mb:.1f} MB)")
    print(f"   Can be deleted to save space, or kept as backup")


def main():
    parser = argparse.ArgumentParser(description="转换 PathMNIST NPZ 为目录结构")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "prepared",
        help="data/prepared 根目录"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有目录"
    )
    args = parser.parse_args()

    convert_pathmnist(args.data_root, args.force)


if __name__ == "__main__":
    main()
