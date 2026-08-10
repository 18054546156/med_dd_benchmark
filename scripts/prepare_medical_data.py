#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗数据三层流水线：官方原始数据 -> prepared 数据 -> DD 算法 loader。

这个脚本不修改 raw/ 下的官方仓库，也不把下载逻辑塞进任何 DD 算法目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'utils') not in sys.path:
    sys.path.insert(0, str(ROOT / 'utils'))

from medical_dataset_utils import MEDICAL_DATASET_SPECS  # noqa: E402


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
COVID_CLASSES = ('COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia')
KVASIR_CLASSES = (
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis',
)

OFFICIAL_SOURCES = {
    'PathMNIST': {
        'page': 'https://github.com/MedMNIST/MedMNIST',
        'download': 'https://zenodo.org/records/10519652/files/pathmnist.npz?download=1',
    },
    'COVID': {
        'page': 'https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database',
        'download': 'kaggle datasets download -d tawsifurrahman/covid19-radiography-database',
        'dataset_id': 'tawsifurrahman/covid19-radiography-database',
    },
    'Kvasir': {
        'page': 'https://datasets.simula.no/kvasir/',
        'download': 'https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip',
    },
}


def md5sum(path: Path) -> str:
    """计算官方文件 MD5，用于下载完整性校验。"""
    digest = hashlib.md5()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256sum(path: Path) -> str:
    """计算压缩包 SHA-256，记录下载文件的实际版本。"""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_dirs(data_root: Path, dataset: str) -> tuple[Path, Path]:
    """返回 raw 和 prepared 的标准目录。"""
    return data_root / 'raw' / dataset, data_root / 'prepared' / dataset


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    """避免在未确认时覆盖已有 prepared 数据。"""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f'目标目录非空，请使用 --overwrite: {path}')
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def verify_pathmnist(raw_dir: Path) -> Path:
    """检查 MedMNIST 官方 NPZ 的 keys、shape 和 MD5。"""
    npz_path = raw_dir / 'pathmnist.npz'
    if not npz_path.exists():
        raise FileNotFoundError(
            f'未找到 {npz_path}。先运行 download PathMNIST，或放入官方 pathmnist.npz。'
        )

    expected_md5 = 'a8b06965200029087d5bd730944a56c1'
    actual_md5 = md5sum(npz_path)
    if actual_md5 != expected_md5:
        raise ValueError(f'PathMNIST MD5 不匹配: {actual_md5} != {expected_md5}')

    import numpy as np

    with np.load(npz_path, allow_pickle=False) as archive:
        expected_keys = {
            'train_images', 'val_images', 'test_images',
            'train_labels', 'val_labels', 'test_labels',
        }
        if set(archive.files) != expected_keys:
            raise ValueError(f'PathMNIST keys 不符合官方格式: {archive.files}')
        if archive['train_images'].shape[1:] != (28, 28, 3):
            raise ValueError(f'PathMNIST 图像 shape 异常: {archive["train_images"].shape}')
        if archive['train_labels'].shape[1:] != (1,):
            raise ValueError(f'PathMNIST 标签 shape 异常: {archive["train_labels"].shape}')

    return npz_path


def download_pathmnist(data_root: Path) -> None:
    """调用官方 MedMNIST loader 下载，不手写镜像地址。"""
    raw_dir, _ = dataset_dirs(data_root, 'PathMNIST')
    raw_dir.mkdir(parents=True, exist_ok=True)

    from medmnist import PathMNIST

    # MedMNIST 会按官方文件名下载并执行自身的完整性校验。
    PathMNIST(split='train', root=str(raw_dir), download=True)
    npz_path = verify_pathmnist(raw_dir)
    write_download_record(
        raw_dir,
        {
            'dataset': 'PathMNIST',
            'contract': MEDICAL_DATASET_SPECS['PathMNIST'],
            'source': OFFICIAL_SOURCES['PathMNIST'],
            'file': str(npz_path.relative_to(raw_dir)),
            'md5': md5sum(npz_path),
        },
    )
    print(f'PathMNIST 官方文件已验证: {npz_path}')
    print(f'来源: {OFFICIAL_SOURCES["PathMNIST"]["download"]}')


def write_download_record(raw_dir: Path, payload: dict) -> None:
    """在 raw 层保存下载来源，避免 prepared 结果失去来源信息。"""
    (raw_dir / 'download.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def prepare_download_dir(raw_dir: Path, overwrite: bool) -> Path:
    """准备单个数据集的 raw 目录，避免无意覆盖已有官方下载文件。"""
    source_dir = raw_dir / 'source'
    if source_dir.exists() and any(source_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f'原始解压目录非空，请使用 --overwrite: {source_dir}'
            )
        shutil.rmtree(source_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return source_dir


def safe_extract(zip_path: Path, target_dir: Path) -> None:
    """安全解压官方 zip，拒绝跳出目标目录的路径。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            destination = (target_dir / member.filename).resolve()
            if destination != target_root and target_root not in destination.parents:
                raise ValueError(f'压缩包包含越界路径: {member.filename}')
        archive.extractall(target_dir)


def download_url(url: str, target: Path) -> None:
    """下载公开官方文件，不改变文件内容。"""
    request = Request(url, headers={'User-Agent': 'med-dd-benchmark/1.0'})
    with urlopen(request) as response, target.open('wb') as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def download_covid(data_root: Path, overwrite: bool) -> None:
    """通过 Kaggle 官方 CLI 下载 COVID-19 Radiography Database。"""
    raw_dir = data_root / 'raw' / 'COVID'
    source_dir = prepare_download_dir(raw_dir, overwrite)
    try:
        subprocess.run(
            [
                'kaggle', 'datasets', 'download',
                '-d', OFFICIAL_SOURCES['COVID']['dataset_id'],
                '-p', str(raw_dir),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            '未找到 kaggle 命令。请先安装 kaggle 并配置 Kaggle API 凭据，'
            '再重新运行 download --dataset COVID。'
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            'Kaggle 下载失败。请检查 API 凭据、网络和数据集访问权限。'
        ) from exc

    archives = sorted(raw_dir.glob('*.zip'), key=lambda path: path.stat().st_mtime)
    if not archives:
        raise FileNotFoundError(f'Kaggle 下载完成但未找到 zip: {raw_dir}')
    archive = archives[-1]
    safe_extract(archive, source_dir)
    write_download_record(
        raw_dir,
        {
            'dataset': 'COVID',
            'source': OFFICIAL_SOURCES['COVID'],
            'archive': archive.name,
            'sha256': sha256sum(archive),
            'extracted_dir': str(source_dir.relative_to(data_root)),
        },
    )
    print(f'COVID 官方压缩包已解压: {source_dir}')


def download_kvasir(data_root: Path, overwrite: bool) -> None:
    """下载并解压 Simula 官方 Kvasir v2 压缩包。"""
    raw_dir = data_root / 'raw' / 'Kvasir'
    source_dir = prepare_download_dir(raw_dir, overwrite)
    archive = raw_dir / 'kvasir-dataset-v2.zip'
    if archive.exists() and not overwrite:
        raise FileExistsError(f'压缩包已存在，请使用 --overwrite: {archive}')
    print(f'开始下载 Kvasir v2: {OFFICIAL_SOURCES["Kvasir"]["download"]}')
    download_url(OFFICIAL_SOURCES['Kvasir']['download'], archive)
    safe_extract(archive, source_dir)
    write_download_record(
        raw_dir,
        {
            'dataset': 'Kvasir',
            'version': 'v2',
            'source': OFFICIAL_SOURCES['Kvasir'],
            'archive': archive.name,
            'sha256': sha256sum(archive),
            'extracted_dir': str(source_dir.relative_to(data_root)),
        },
    )
    print(f'Kvasir v2 官方压缩包已解压: {source_dir}')


def copy_or_link(source: Path, target: Path, mode: str) -> None:
    """把官方 NPZ 放到 prepared 层，默认复制以保证层间独立。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'hardlink':
        try:
            target.hardlink_to(source)
            return
        except OSError:
            print('硬链接不可用，回退为普通复制。')
    shutil.copy2(source, target)


def prepare_pathmnist(data_root: Path, materialize: str, overwrite: bool) -> None:
    """复制官方 NPZ 到 prepared，并生成来源 manifest。"""
    raw_dir, prepared_dir = dataset_dirs(data_root, 'PathMNIST')
    source = verify_pathmnist(raw_dir)
    ensure_empty_or_overwrite(prepared_dir, overwrite)
    target = prepared_dir / source.name
    copy_or_link(source, target, materialize)
    write_manifest(
        prepared_dir,
        {
            'dataset': 'PathMNIST',
            'source': OFFICIAL_SOURCES['PathMNIST'],
            'raw_file': str(source.relative_to(data_root)),
            'prepared_file': str(target.relative_to(data_root)),
            'format': 'MedMNIST NPZ; loader performs 28x28 -> 32x32 resize',
            'md5': md5sum(target),
        },
    )
    print(f'PathMNIST prepared 完成: {target}')


def normalize_source_class(name: str, dataset: str) -> str | None:
    """把官方下载目录名映射到项目固定类别名。"""
    normalized = name.strip().replace('_', ' ').lower()
    if dataset == 'COVID':
        aliases = {
            'covid': 'COVID',
            'lung opacity': 'Lung_Opacity',
            'normal': 'Normal',
            'viral pneumonia': 'Viral_Pneumonia',
        }
        return aliases.get(normalized)
    if dataset == 'Kvasir':
        return next((item for item in KVASIR_CLASSES if item.lower() == normalized), None)
    raise ValueError(f'ImageFolder 数据集不支持: {dataset}')


def collect_class_images(source_dir: Path, dataset: str) -> dict[str, list[Path]]:
    """从官方原始目录递归收集图片，并确认每个类别都存在。"""
    expected = COVID_CLASSES if dataset == 'COVID' else KVASIR_CLASSES
    collected = {name: [] for name in expected}
    for directory in (path for path in source_dir.rglob('*') if path.is_dir()):
        class_name = normalize_source_class(directory.name, dataset)
        if class_name is None:
            continue

        # COVID 官方压缩包同时包含 images/ 和 masks/；分类任务只能读取 images/。
        image_roots = [
            child for child in directory.iterdir()
            if child.is_dir() and child.name.lower() in {'image', 'images'}
        ]
        if not image_roots:
            image_roots = [directory]
        for image_root in image_roots:
            collected[class_name].extend(
                path for path in image_root.rglob('*')
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    missing = [name for name, files in collected.items() if not files]
    if missing:
        raise FileNotFoundError(f'{dataset} 缺少类别或图片: {missing}; source={source_dir}')
    return {name: sorted(set(files)) for name, files in collected.items()}


def materialize_image(source: Path, target: Path) -> dict:
    """复制图片并统一转为 RGB，记录尺寸和原始相对路径。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        original_mode = image.mode
        if image.mode == 'RGB':
            shutil.copy2(source, target)
        else:
            # 医疗 X 光图常见灰度模式，统一为三通道以匹配 ConvNet 输入合同。
            image.convert('RGB').save(target)
        with Image.open(target) as converted:
            width, height = converted.size
    return {
        'source': str(source),
        'target': str(target),
        'original_mode': original_mode,
        'mode': 'RGB',
        'width': width,
        'height': height,
    }


def prepare_imagefolder(
    data_root: Path,
    dataset: str,
    source_dir: Path,
    seed: int,
    test_ratio: float,
    overwrite: bool,
) -> None:
    """按类别固定随机划分并输出 ImageFolder train/test。"""
    if not 0 < test_ratio < 1:
        raise ValueError('--test-ratio 必须在 0 和 1 之间')
    if not source_dir.exists():
        raise FileNotFoundError(f'原始解压目录不存在: {source_dir}')

    _, prepared_dir = dataset_dirs(data_root, dataset)
    ensure_empty_or_overwrite(prepared_dir, overwrite)
    class_images = collect_class_images(source_dir, dataset)
    manifest_files = []
    counts = Counter()

    for class_index, (class_name, files) in enumerate(class_images.items()):
        rng = random.Random(seed + class_index)
        shuffled = list(files)
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_ratio))
        split_files = {'test': shuffled[:test_count], 'train': shuffled[test_count:]}
        if not split_files['train']:
            raise ValueError(f'{dataset}/{class_name} 图片太少，无法保留 train 样本')

        for split, split_sources in split_files.items():
            for source in split_sources:
                target_name = source.name
                target = prepared_dir / split / class_name / target_name
                if target.exists():
                    target = target.with_name(f'{source.stem}_{hashlib.md5(str(source).encode()).hexdigest()[:8]}{source.suffix}')
                record = materialize_image(source, target)
                # manifest 使用相对路径，避免换机器后绝对路径失效。
                record['source'] = str(source.relative_to(source_dir))
                record['target'] = str(target.relative_to(prepared_dir))
                record.update({
                    'dataset': dataset,
                    'split': split,
                    'class_name': class_name,
                    'class_index': class_index,
                })
                manifest_files.append(record)
                counts[f'{split}/{class_name}'] += 1

    write_manifest(
        prepared_dir,
        {
            'dataset': dataset,
            'contract': MEDICAL_DATASET_SPECS[dataset],
            'source': OFFICIAL_SOURCES[dataset],
            'source_dir': str(source_dir),
            'seed': seed,
            'test_ratio': test_ratio,
            'classes': list(class_images),
            'counts': dict(counts),
            'files': manifest_files,
        },
    )
    print(f'{dataset} prepared 完成: {prepared_dir}')
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))


def write_manifest(prepared_dir: Path, payload: dict) -> None:
    """统一写出可追溯的 JSON manifest。"""
    manifest_path = prepared_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def validate_imagefolder(prepared_dir: Path, dataset: str) -> None:
    """验证 prepared ImageFolder 的类别、图片模式和 train/test 非空。"""
    expected = COVID_CLASSES if dataset == 'COVID' else KVASIR_CLASSES
    for split in ('train', 'test'):
        for class_name in expected:
            class_dir = prepared_dir / split / class_name
            if not class_dir.exists():
                raise ValueError(f'缺少类别目录: {class_dir}')
            files = [path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
            if not files:
                raise ValueError(f'空类别目录: {class_dir}')
            for image_path in files:
                with Image.open(image_path) as image:
                    if image.mode != 'RGB':
                        raise ValueError(f'图片不是 RGB: {image_path} ({image.mode})')
    print(f'{dataset} prepared 验证通过: {prepared_dir}')


def validate(data_root: Path, dataset: str) -> None:
    """执行 prepared 层的离线验证，不触发下载。"""
    _, prepared_dir = dataset_dirs(data_root, dataset)
    if dataset == 'PathMNIST':
        verify_pathmnist(prepared_dir)
        if not (prepared_dir / 'manifest.json').exists():
            raise FileNotFoundError(f'缺少 manifest.json: {prepared_dir}')
        print(f'PathMNIST prepared 验证通过: {prepared_dir}')
    else:
        validate_imagefolder(prepared_dir, dataset)
        if not (prepared_dir / 'manifest.json').exists():
            raise FileNotFoundError(f'缺少 manifest.json: {prepared_dir}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='官方医疗数据 -> prepared -> DD loader 流水线')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data', help='raw/prepared 的共同根目录')
    subparsers = parser.add_subparsers(dest='command', required=True)

    download = subparsers.add_parser('download', help='从官方来源下载并解压原始数据')
    download.add_argument(
        '--dataset', choices=['PathMNIST', 'COVID', 'Kvasir'], required=True
    )
    download.add_argument('--overwrite', action='store_true')

    prepare = subparsers.add_parser('prepare', help='把 raw 数据整理成 prepared 格式')
    prepare.add_argument('--dataset', choices=['PathMNIST', 'COVID', 'Kvasir'], required=True)
    prepare.add_argument('--source-dir', type=Path, help='COVID/Kvasir 官方解压目录')
    prepare.add_argument('--seed', type=int, default=20260810)
    prepare.add_argument('--test-ratio', type=float, default=0.2)
    prepare.add_argument('--materialize', choices=['copy', 'hardlink'], default='copy')
    prepare.add_argument('--overwrite', action='store_true')

    validate_parser = subparsers.add_parser('validate', help='离线验证 prepared 数据和 manifest')
    validate_parser.add_argument('--dataset', choices=['PathMNIST', 'COVID', 'Kvasir'], required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.resolve()
    if args.command == 'download':
        if args.dataset == 'PathMNIST':
            download_pathmnist(data_root)
        elif args.dataset == 'COVID':
            download_covid(data_root, args.overwrite)
        else:
            download_kvasir(data_root, args.overwrite)
    elif args.command == 'prepare':
        if args.dataset == 'PathMNIST':
            prepare_pathmnist(data_root, args.materialize, args.overwrite)
        else:
            if args.source_dir is None:
                raise ValueError('COVID/Kvasir prepare 必须提供 --source-dir')
            prepare_imagefolder(
                data_root,
                args.dataset,
                args.source_dir.resolve(),
                args.seed,
                args.test_ratio,
                args.overwrite,
            )
    elif args.command == 'validate':
        validate(data_root, args.dataset)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
