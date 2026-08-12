#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NCFM重复Normalize问题修复方案

问题诊断:
- medical_dataset_utils.py 在加载时已做 Normalize
- NCFM 的 diffaug.py 训练时又做一次 Normalize
- 导致训练数据被重复归一化，验证数据只归一化一次
- 结果: Train acc 84%, Val acc 52% (异常)

修复方案A (推荐): 为NCFM提供不带Normalize的数据
修复方案B: 让NCFM检测并跳过已Normalize的数据
"""

# ============================================================================
# 方案A: 修改 medical_dataset_utils.py (推荐)
# ============================================================================

# 在 medical_dataset_utils.py 中添加参数控制是否Normalize

# 修改 _medical_transform 函数签名:
def _medical_transform_A(dataset_name, use_zca=False, skip_normalize=False):
    """创建所有算法共用的医疗数据变换；ZCA 时跳过普通 Normalize。

    Args:
        dataset_name: 数据集名称
        use_zca: 是否使用ZCA白化
        skip_normalize: 是否跳过Normalize (NCFM需要设为True)
    """
    spec = get_medical_spec(dataset_name)
    resize = transforms.Resize(
        spec['im_size'], interpolation=transforms.InterpolationMode.BICUBIC
    )
    steps = [transforms.ToTensor(), resize]
    if not use_zca and not skip_normalize:
        steps.append(transforms.Normalize(spec['mean'], spec['std']))
    return transforms.Compose(steps)


# 修改 load_medical_splits 函数签名:
def load_medical_splits_A(dataset_name, data_path, use_zca=False, skip_normalize=False):
    """读取共享的 train/val/test 三个 split。

    Args:
        dataset_name: 数据集名称
        data_path: 数据根目录
        use_zca: 是否使用ZCA白化
        skip_normalize: 是否跳过Normalize (NCFM需要设为True)
    """
    spec = get_medical_spec(dataset_name)
    root = resolve_medical_data_root(data_path, dataset_name)
    transform = _medical_transform(dataset_name, use_zca=use_zca, skip_normalize=skip_normalize)
    # ... 其余代码不变


# 在 adapted/ncfm/utils/utils.py 的 _load_medical_dataset 中:
def _load_medical_dataset_A(dataset, data_dir, size, evaluation_split="val"):
    """加载医疗数据集（PathMNIST/COVID/Kvasir）。

    NCFM 使用 diffaug 进行训练时 Normalize，因此加载数据时跳过 Normalize。
    """
    spec_name = {
        "pathmnist": "PathMNIST",
        "covid": "COVID",
        "kvasir": "Kvasir",
    }[dataset]
    spec = MEDICAL_DATASET_SPECS[spec_name]

    # NCFM 的 diffaug 会在训练时做 Normalize，所以这里跳过
    splits = load_medical_splits(spec_name, data_dir, skip_normalize=True)

    train_dataset = splits["train"]
    if evaluation_split not in {"val", "test"}:
        raise ValueError(f"不支持的医疗评估 split: {evaluation_split}")
    val_dataset = splits[evaluation_split]

    return (
        _attach_dataset_metadata(train_dataset, spec["num_classes"]),
        _attach_dataset_metadata(val_dataset, spec["num_classes"]),
    )


# ============================================================================
# 方案B: 修改 NCFM 的 diffaug.py (不推荐，侵入性更大)
# ============================================================================

# 在 adapted/ncfm/utils/diffaug.py 中添加检测逻辑:

def diffaug_B(args, device="cuda", skip_normalize_on_medical=True):
    """Differentiable augmentation for condensation

    Args:
        args: 参数对象
        device: 设备
        skip_normalize_on_medical: 对医疗数据集跳过Normalize（因为已经在加载时做了）
    """
    aug_type = args.aug_type

    # 检测是否为医疗数据集
    is_medical = args.dataset.lower() in ("pathmnist", "covid", "kvasir")

    if is_medical and skip_normalize_on_medical:
        # 医疗数据集已经在 medical_dataset_utils 中 Normalize 过了
        # 这里只做数据增强，不再 Normalize
        if args.rank == 0:
            print(f"Medical dataset {args.dataset} detected, skipping duplicate Normalize")
        augment = DiffAug(strategy=aug_type, batch=True)
        aug_batch = augment  # 直接使用增强，不加 normalize

        if args.mixup == "cut":
            aug_type = remove_aug(aug_type, "cutout")
        if args.rank == 0:
            print("Augmentataion Net update: ", aug_type)
        augment_rand = DiffAug(strategy=aug_type, batch=False)
        aug_rand = augment_rand  # 直接使用增强，不加 normalize
    else:
        # CIFAR10 等数据集保持原有逻辑
        normalize = Normalize(
            mean=MEANS[args.dataset], std=STDS[args.dataset], device=device
        )
        if args.rank == 0:
            print("Augmentataion Matching: ", aug_type)
        augment = DiffAug(strategy=aug_type, batch=True)
        aug_batch = transforms.Compose([normalize, augment])

        if args.mixup == "cut":
            aug_type = remove_aug(aug_type, "cutout")
        if args.rank == 0:
            print("Augmentataion Net update: ", aug_type)
        augment_rand = DiffAug(strategy=aug_type, batch=False)
        aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


# ============================================================================
# 推荐实施方案A的原因
# ============================================================================

"""
1. 更清晰的责任分离:
   - medical_dataset_utils: 负责数据加载，可选是否Normalize
   - NCFM diffaug: 负责训练时增强和Normalize

2. 不影响其他算法:
   - 其他算法继续使用默认 skip_normalize=False
   - 只有NCFM需要传 skip_normalize=True

3. 更容易验证:
   - 可以在加载时直接检查数据范围
   - 避免在训练中才发现问题

4. 更符合NCFM原始设计:
   - NCFM原本就期望输入是[0,1]范围的数据
   - diffaug负责Normalize到标准分布

5. 测试验证简单:
   - 检查train数据: 应该在[0,1]范围
   - 检查val数据: 也应该在[0,1]范围
   - diffaug会将两者都Normalize到标准分布
"""

# ============================================================================
# 实施步骤 (方案A)
# ============================================================================

"""
1. 修改 utils/medical_dataset_utils.py:
   - 在 _medical_transform() 添加 skip_normalize 参数
   - 在 load_medical_splits() 添加 skip_normalize 参数

2. 修改 adapted/ncfm/utils/utils.py:
   - 在 _load_medical_dataset() 中调用时传 skip_normalize=True

3. 停止当前HPC作业 (如果仍在运行)

4. 测试修复:
   python -c "
   from utils.medical_dataset_utils import load_medical_splits
   splits = load_medical_splits('PathMNIST', 'data/prepared', skip_normalize=True)
   img, _ = splits['train'][0]
   print(f'Min: {img.min():.4f}, Max: {img.max():.4f}')
   # 应该输出: Min: 0.0000, Max: 1.0000 (或接近)
   "

5. 重新运行NCFM pretrain:
   python scripts/run_config.py \
     --config configs/ncfm/pathmnist/ipc10_full.yaml \
     --algorithm ncfm \
     --stage pretrain \
     --run

6. 验证结果:
   - Train acc 应该达到 88-90%
   - Val acc 应该达到 85-88%
   - 两者差距应该在合理范围（3-5%）
"""

# ============================================================================
# 预期修复后的结果
# ============================================================================

"""
修复前 (当前):
  Train acc: 84.3-84.8%
  Val acc:   52.0-53.1%  ← 异常低
  Gap:       ~32%        ← 异常大

修复后 (预期):
  Train acc: 88-90%
  Val acc:   85-88%
  Gap:       3-5%        ← 正常范围

参考 HoP buffer:
  Train acc: 93.7-94.3%
  Test acc:  86.7-90.4%
  Gap:       ~4%         ← 正常水平
"""
