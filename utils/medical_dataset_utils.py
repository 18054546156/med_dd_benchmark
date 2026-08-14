"""
医疗数据集工具函数
提供MedMNIST标签标量化、数据集验证等通用功能
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 三个医疗数据集的固定属性，供各算法和测试脚本共享。
MEDICAL_DATASET_SPECS = {
    # PathMNIST 的类别和通道来自 MedMNIST；32x32 与 mean/std 是本 benchmark 约定。
    'PathMNIST': {
        'channel': 3,
        'im_size': (32, 32),
        'num_classes': 9,
        'mean': [0.741, 0.533, 0.706],
        'std': [0.402, 0.821, 0.407],
        'format': 'MedMNIST',
    },
    # COVID/Kvasir 的 mean/std 是 ImageNet 预训练约定，不是官方数据统计量。
    'COVID': {
        'channel': 3,
        'im_size': (112, 112),
        'num_classes': 4,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'format': 'ImageFolder',
    },
    'Kvasir': {
        'channel': 3,
        'im_size': (128, 128),
        'num_classes': 8,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'format': 'ImageFolder',
    },
}


def get_medical_spec(dataset_name):
    """返回统一规范中的数据集属性，未知名称直接报错。"""
    try:
        return MEDICAL_DATASET_SPECS[dataset_name]
    except KeyError as exc:
        supported = ', '.join(MEDICAL_DATASET_SPECS)
        raise ValueError(f'不支持的数据集: {dataset_name}; 可选值: {supported}') from exc


def get_medical_statistics(dataset_name, data_path):
    """读取 prepared 数据的 train-only 统计量。

    ``statistics.json`` 是正式实验的权威来源；固定 spec 仅作为旧 smoke
    数据布局的兼容 fallback，不能作为正式结果的统计量证据。
    """
    spec = get_medical_spec(dataset_name)
    root = resolve_medical_data_root(data_path, dataset_name)
    path = root / 'statistics.json'
    if not path.is_file():
        return list(spec['mean']), list(spec['std'])

    import json

    payload = json.loads(path.read_text(encoding='utf-8'))
    stats = payload.get('statistics', payload)
    mean = stats.get('mean')
    std = stats.get('std')
    if (not isinstance(mean, list) or not isinstance(std, list)
            or len(mean) != 3 or len(std) != 3
            or any(float(value) <= 0 for value in std)):
        raise ValueError(f'无效的 train-only 统计量: {path}')
    return [float(value) for value in mean], [float(value) for value in std]


def get_medmnist_root(data_path):
    """
    返回 PathMNIST 的统一存储目录。

    新目录使用 ``data/PathMNIST``；如果传入目录已经直接包含
    ``pathmnist.npz``，则保留旧布局，避免重复下载已有缓存。
    """
    base = resolve_medical_data_root(data_path, 'PathMNIST')
    if (base / 'pathmnist.npz').exists():
        return str(base)

    root = base / 'PathMNIST'
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def resolve_medical_data_root(data_path, dataset_name):
    """解析共享 data/prepared 根目录，兼容从仓库根目录或算法子目录启动。

    算法的原始代码普遍把 ``data_path`` 当作当前工作目录的相对路径。
    这里统一尝试仓库根目录、当前目录和直接传入目录，避免不同算法读到
    不同的副本。
    """
    dataset_name = str(dataset_name)
    folder_name = dataset_name if dataset_name == 'PathMNIST' else dataset_name
    requested = Path(data_path).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend((Path.cwd() / requested, PROJECT_ROOT / requested))

    for candidate in candidates:
        candidate = candidate.resolve()
        direct = candidate / folder_name
        prepared = candidate / 'prepared' / folder_name
        # 当 data/PathMNIST 和 data/prepared/PathMNIST 同时存在时，
        # 优先使用统一 prepared 层，避免不同算法读到不同副本。
        if (prepared / 'train').exists() or (prepared / 'pathmnist.npz').exists():
            return prepared
        if (direct / 'train').exists() or (direct / 'pathmnist.npz').exists():
            return direct
        if (candidate / 'pathmnist.npz').exists() and dataset_name == 'PathMNIST':
            return candidate

    # 让调用方收到包含实际路径的错误，而不是后面出现难以定位的 FileNotFoundError。
    return (PROJECT_ROOT / 'data' / 'prepared' / folder_name).resolve()


def _medical_transform(dataset_name, use_zca=False, skip_normalize=False,
                       mean=None, std=None):
    """创建医疗数据变换；可为 train 保留 raw [0, 1] 数据。"""
    spec = get_medical_spec(dataset_name)
    mean = spec['mean'] if mean is None else mean
    std = spec['std'] if std is None else std
    resize = transforms.Resize(
        spec['im_size'], interpolation=transforms.InterpolationMode.BICUBIC
    )
    steps = [transforms.ToTensor(), resize]
    if not use_zca and not skip_normalize:
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def load_medical_splits(dataset_name, data_path, use_zca=False,
                        train_skip_normalize=False):
    """读取共享的 train/val/test 三个 split。

    返回字典而不是某个算法专用的 tuple，算法适配器可以按自己的返回合同
    取用 train、val 和 test；这样不会在每个算法里重新切分数据。
    ``train_skip_normalize=True`` 仅让 train 保持 raw [0, 1]，
    val/test 仍使用同一套 train-only 统计量。
    """
    spec = get_medical_spec(dataset_name)
    root = resolve_medical_data_root(data_path, dataset_name)
    mean, std = get_medical_statistics(dataset_name, data_path)
    train_transform = _medical_transform(
        dataset_name,
        use_zca=use_zca,
        skip_normalize=train_skip_normalize,
        mean=mean,
        std=std,
    )
    eval_transform = _medical_transform(
        dataset_name,
        use_zca=use_zca,
        mean=mean,
        std=std,
    )

    if spec['format'] == 'MedMNIST':
        from medmnist import PathMNIST

        # root 已经是 prepared/PathMNIST；MedMNIST 会在该目录查找 NPZ。
        med_root = str(root)
        splits = {
            split: MedMNISTWrapper(
                PathMNIST(
                    split=split,
                    root=med_root,
                    download=True,
                    transform=train_transform if split == 'train' else eval_transform,
                )
            )
            for split in ('train', 'val', 'test')
        }
        return splits

    if not (root / 'train').exists() or not (root / 'test').exists():
        raise FileNotFoundError(
            f'{dataset_name} 缺少共享 train/test 目录: {root}'
        )
    val_root = root / 'val'
    if not val_root.exists():
        raise FileNotFoundError(
            f'{dataset_name} 缺少共享 val 目录: {val_root}; '
            '请先运行 scripts/prepare_medical_data.py prepare'
        )

    return {
        split: datasets.ImageFolder(
            root / split,
            transform=train_transform if split == 'train' else eval_transform,
        )
        for split in ('train', 'val', 'test')
    }


def get_split_counts(dataset_name, data_path):
    """从 manifest 或目录统计每个 split，供审计和报告使用。"""
    root = resolve_medical_data_root(data_path, dataset_name)
    manifest = root / 'manifest.json'
    if manifest.exists():
        import json

        payload = json.loads(manifest.read_text(encoding='utf-8'))
        if 'counts' in payload:
            return payload['counts']

    counts = {}
    for split in ('train', 'val', 'test'):
        split_root = root / split
        if split_root.exists():
            for class_dir in split_root.iterdir():
                if class_dir.is_dir():
                    counts[f'{split}/{class_dir.name}'] = sum(
                        1 for item in class_dir.iterdir() if item.is_file()
                    )
    return counts


def scalarize_label(label):
    """把 MedMNIST/ImageFolder 标签统一为 Python ``int``。"""
    if torch.is_tensor(label):
        if label.numel() != 1:
            raise ValueError(f'标签必须只有一个元素，当前 shape={tuple(label.shape)}')
        return int(label.detach().cpu().reshape(-1)[0].item())

    array = np.asarray(label)
    if array.size != 1:
        raise ValueError(f'标签必须只有一个元素，当前 shape={array.shape}')
    return int(array.reshape(-1)[0])


class MedMNISTWrapper(Dataset):
    """
    MedMNIST标签标量化包装器

    问题：MedMNIST返回的标签是 np.ndarray shape (1,)，例如 array([5])
    解决：将标签转换为标量整数，例如 5

    使用方法：
        from medmnist import PathMNIST
        raw_dataset = PathMNIST(split='train', download=True)
        dataset = MedMNISTWrapper(raw_dataset)
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # 标签标量化：array([5]) -> 5，保证主循环和 DataLoader 行为一致。
        return img, scalarize_label(label)

    def __getattr__(self, name):
        """把 classes、labels 等属性透传给底层 MedMNIST 数据集。"""
        if name == 'dataset':
            raise AttributeError(name)
        if name == 'targets':
            # torchvision/ImageFolder uses targets; MedMNIST uses labels.
            labels = getattr(self.dataset, 'labels', None)
            if labels is not None:
                return [scalarize_label(label) for label in labels]
        return getattr(self.dataset, name)


def validate_dataset_contract(dataset_name, dst_train, dst_test,
                              num_classes, channel, im_size, batch_size=4):
    """
    验证数据集是否符合统一规范

    Args:
        dataset_name: 数据集名称
        dst_train: 训练集
        dst_test: 测试集
        num_classes: 类别数
        channel: 通道数
        im_size: 图像尺寸 (H, W)
        batch_size: 验证批大小

    Returns:
        bool: 是否通过验证
    """

    print(f"\n{'='*60}")
    print(f"验证数据集: {dataset_name}")
    print(f"{'='*60}")

    try:
        # 验证1: 训练集标签格式
        print("✓ 检查训练集标签格式...")
        img, label = dst_train[0]

        # 标签必须是标量
        label_value = scalarize_label(label)

        print(f"  训练集首个标签: {label_value} (类型: {type(label).__name__})")

        # 验证2: 测试集标签格式
        print("✓ 检查测试集标签格式...")
        img, label = dst_test[0]

        label_value = scalarize_label(label)

        print(f"  测试集首个标签: {label_value} (类型: {type(label).__name__})")

        # 验证3: 图像形状
        print("✓ 检查图像形状...")
        img, _ = dst_train[0]
        if isinstance(img, torch.Tensor):
            actual_shape = img.shape
        else:
            actual_shape = np.array(img).shape

        expected_shape = (channel, im_size[0], im_size[1])
        assert actual_shape == expected_shape, \
            f"图像形状错误: {actual_shape} vs 期望 {expected_shape}"
        print(f"  图像形状: {actual_shape}")

        # 验证4: 标签范围
        print("✓ 检查标签范围...")
        sample_size = min(100, len(dst_train))
        labels = []
        for i in range(sample_size):
            _, label = dst_train[i]
            labels.append(scalarize_label(label))

        min_label = min(labels)
        max_label = max(labels)
        assert min_label >= 0, f"标签最小值 {min_label} < 0"
        assert max_label < num_classes, f"标签最大值 {max_label} >= {num_classes}"
        print(f"  标签范围: [{min_label}, {max_label}] (期望: [0, {num_classes-1}])")

        # 验证5: DataLoader兼容性
        print("✓ 检查DataLoader兼容性...")
        from torch.utils.data import DataLoader
        loader = DataLoader(dst_train, batch_size=batch_size, shuffle=False)
        images, labels = next(iter(loader))

        assert images.shape == (batch_size, channel, im_size[0], im_size[1]), \
            f"批次图像形状错误: {images.shape}"
        assert labels.shape == (batch_size,), \
            f"批次标签形状错误: {labels.shape}，应为 ({batch_size},)"
        assert labels.dtype == torch.long, \
            f"批次标签类型错误: {labels.dtype}，应为 torch.long"

        print(f"  批次图像形状: {images.shape}")
        print(f"  批次标签形状: {labels.shape}, dtype: {labels.dtype}")

        print(f"\n{'='*60}")
        print(f"✅ {dataset_name} 验证通过！")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ {dataset_name} 验证失败！")
        print(f"错误: {str(e)}")
        print(f"{'='*60}\n")
        raise


def smoke_test_dataset(dataset_name, get_dataset_fn, data_path):
    """
    快速冒烟测试：只加载首个batch验证数据加载

    Args:
        dataset_name: 数据集名称
        get_dataset_fn: get_dataset函数
        data_path: 数据路径

    Returns:
        bool: 是否通过测试
    """

    print(f"\n{'='*60}")
    print(f"冒烟测试: {dataset_name}")
    print(f"{'='*60}")

    try:
        # 调用get_dataset获取数据
        result = get_dataset_fn(dataset_name, data_path)

        # 不同算法末尾可能附带 zca 或按类 loader，只读取统一合同的前九项。
        if len(result) < 9:
            raise ValueError(f'get_dataset 返回值不足九项: {len(result)}')
        channel, im_size, num_classes, class_names, mean, std, \
            dst_train, dst_test, testloader = result[:9]

        print(f"✓ 数据集属性:")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 通道数: {channel}")
        print(f"  - 图像尺寸: {im_size}")
        print(f"  - 训练集大小: {len(dst_train)}")
        print(f"  - 测试集大小: {len(dst_test)}")

        # 加载首个batch
        print(f"\n✓ 加载首个batch...")
        images, labels = next(iter(testloader))

        print(f"  - 图像形状: {images.shape}")
        print(f"  - 标签形状: {labels.shape}")
        print(f"  - 图像dtype: {images.dtype}")
        print(f"  - 标签dtype: {labels.dtype}")
        print(f"  - 标签样例: {labels[:min(5, len(labels))].tolist()}")

        # 基本验证
        assert images.dtype == torch.float32
        assert labels.dtype == torch.long
        assert labels.dim() == 1  # 标签必须是1维
        assert labels.min() >= 0
        assert labels.max() < num_classes

        print(f"\n✅ {dataset_name} 冒烟测试通过！")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n❌ {dataset_name} 冒烟测试失败！")
        print(f"错误: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return False


# 类别名称映射
CLASS_NAMES = {
    'PathMNIST': [
        'adipose',
        'background',
        'debris',
        'lymphocytes',
        'mucus',
        'smooth_muscle',
        'normal_colon_mucosa',
        'cancer_associated_stroma',
        'colorectal_adenocarcinoma_epithelium'
    ],
    'COVID': [
        'COVID',
        'Lung_Opacity',
        'Normal',
        'Viral_Pneumonia'
    ],
    'Kvasir': [
        'dyed-lifted-polyps',
        'dyed-resection-margins',
        'esophagitis',
        'normal-cecum',
        'normal-pylorus',
        'normal-z-line',
        'polyps',
        'ulcerative-colitis'
    ]
}


def get_class_names(dataset_name):
    """获取数据集的类别名称"""
    return CLASS_NAMES.get(dataset_name, None)
