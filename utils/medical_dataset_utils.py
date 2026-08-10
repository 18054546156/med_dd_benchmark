"""
医疗数据集工具函数
提供MedMNIST标签标量化、数据集验证等通用功能
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


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


def get_medmnist_root(data_path):
    """
    返回 PathMNIST 的统一存储目录。

    新目录使用 ``data/PathMNIST``；如果传入目录已经直接包含
    ``pathmnist.npz``，则保留旧布局，避免重复下载已有缓存。
    """
    base = Path(data_path).expanduser()
    if (base / 'pathmnist.npz').exists():
        return str(base)

    root = base / 'PathMNIST'
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


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
