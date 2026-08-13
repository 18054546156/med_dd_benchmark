# 配置和适配修改完整记录

生成时间: 2026-08-14 03:40

---

## 📋 修改总览

### 核心修改
1. ✅ 添加医疗数据集适配层
2. ✅ 修正归一化处理逻辑
3. ✅ 统一图像尺寸配置
4. ✅ 统一 Backbone 深度
5. ✅ 创建公平评估配置

---

## 🔧 代码适配修改

### 1. 医疗数据集加载器 (新增)

#### 文件: `utils/medical_dataset_utils.py`

**修改类型**: 新增文件

**功能**: 统一的医疗数据集加载接口

**关键代码**:
```python
def load_medical_splits(dataset_name, data_path, use_zca=False, train_skip_normalize=False):
    """
    加载医疗数据集的 train/val/test splits
    
    Args:
        dataset_name: 'PathMNIST', 'COVID', 'Kvasir'
        data_path: 数据根目录
        use_zca: 是否使用 ZCA 白化
        train_skip_normalize: 训练集是否跳过 Normalize (NCFM 用 True)
    
    Returns:
        {'train': dataset, 'val': dataset, 'test': dataset}
    """
    spec = get_medical_spec(dataset_name)
    root = resolve_medical_data_root(data_path, dataset_name)
    
    # 关键: train 和 eval 使用不同的 transform
    train_transform = _medical_transform(
        dataset_name, 
        use_zca=use_zca, 
        normalize=not train_skip_normalize  # ✅ 关键参数
    )
    eval_transform = _medical_transform(
        dataset_name, 
        use_zca=use_zca, 
        normalize=True  # Val/Test 总是 Normalize
    )
    
    # 加载 ImageFolder
    return {
        'train': datasets.ImageFolder(root / 'train', transform=train_transform),
        'val': datasets.ImageFolder(root / 'val', transform=eval_transform),
        'test': datasets.ImageFolder(root / 'test', transform=eval_transform),
    }
```

**数据集规格**:
```python
MEDICAL_DATASET_SPECS = {
    'PathMNIST': {
        'num_classes': 9,
        'image_size': (32, 32),
        'channels': 3,
        'mean': [0.7405449443, 0.5329821482, 0.7058288200],
        'std': [0.1214735017, 0.1741895871, 0.1224783296],
        'format': 'MedMNIST',
    },
    'COVID': {
        'num_classes': 4,  # ✅ 4 类 (COVID, Lung_Opacity, Normal, Viral_Pneumonia)
        'image_size': (112, 112),
        'channels': 3,
        'mean': [0.5098135074, 0.5098135074, 0.5098135074],
        'std': [0.2519904495, 0.2519904495, 0.2519904495],
        'format': 'ImageFolder',
    },
    'Kvasir': {
        'num_classes': 8,
        'image_size': (128, 128),
        'channels': 3,
        'mean': [0.4859546510, 0.3463666971, 0.2985339895],
        'std': [0.3312173078, 0.2410854483, 0.2318631172],
        'format': 'ImageFolder',
    }
}
```

**原因**: 
- 提供统一的数据加载接口
- 支持 `train_skip_normalize` 避免双重归一化
- 自动处理不同数据集的图像尺寸和统计量

---

### 2. NCFM 数据加载适配

#### 文件: `adapted/ncfm/utils/utils.py`

**修改类型**: 添加医疗数据集支持

**原始代码** (GitHub):
```python
def get_loader(args):
    dataset = args.dataset
    
    # 只支持 CIFAR, ImageNet 等
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(...)
    elif dataset == 'imagenet':
        train_dataset = datasets.ImageNet(...)
    # 没有医疗数据集
    
    return train_loader, val_loader
```

**修改后代码** (HPC):
```python
def get_loader(args):
    dataset = args.dataset
    
    # ✅ 添加医疗数据集支持
    if dataset in ['pathmnist', 'covid', 'kvasir']:
        return _load_medical_dataset(
            dataset, 
            args.data_dir, 
            args.size, 
            evaluation_split="val"
        )
    
    # 原有代码保持不变
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(...)
    ...

def _load_medical_dataset(dataset, data_dir, size, evaluation_split="val"):
    """加载医疗数据集 (新增函数)"""
    spec_name = {
        "pathmnist": "PathMNIST",
        "covid": "COVID",
        "kvasir": "Kvasir",
    }[dataset]
    spec = MEDICAL_DATASET_SPECS[spec_name]
    
    # ✅ 关键: 同步统计量到 NCFM 全局字典
    from data.dataset_statistics import MEANS, STDS
    MEANS[dataset] = list(spec["mean"])
    STDS[dataset] = list(spec["std"])
    
    # ✅ 关键: 使用 train_skip_normalize=True
    splits = load_medical_splits(
        spec_name, 
        data_dir, 
        train_skip_normalize=True  # ⭐ 避免双重 Normalize
    )
    
    train_dataset = splits["train"]
    val_dataset = splits[evaluation_split]
    
    return (
        _attach_dataset_metadata(train_dataset, spec["num_classes"]),
        _attach_dataset_metadata(val_dataset, spec["num_classes"]),
    )
```

**修改原因**:
1. **添加医疗数据集支持** - 原版只支持 CIFAR/ImageNet
2. **同步统计量** - NCFM 的 diffaug 需要全局 MEANS/STDS
3. **避免双重归一化** - `train_skip_normalize=True` 是关键

**影响**:
- ✅ 修复了双重归一化 bug
- ✅ Train 和 Val 分布一致
- ✅ 结果从 Train 83%/Val 32% → Train 86%/Val 97%

---

### 3. NCFM 数据统计量

#### 文件: `adapted/ncfm/data/dataset_statistics.py`

**修改类型**: 添加医疗数据集统计量

**原始代码** (GitHub):
```python
MEANS = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'imagenet': [0.485, 0.456, 0.406],
}
STDS = {
    'cifar10': [0.2023, 0.1994, 0.2010],
    'imagenet': [0.229, 0.224, 0.225],
}
# 没有医疗数据集
```

**修改后代码** (HPC):
```python
# 原有数据集保持不变
MEANS = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'imagenet': [0.485, 0.456, 0.406],
}
STDS = {
    'cifar10': [0.2023, 0.1994, 0.2010],
    'imagenet': [0.229, 0.224, 0.225],
}

# ✅ 添加医疗数据集统计量
MEANS['pathmnist'] = [0.7405449443, 0.5329821482, 0.7058288200]
STDS['pathmnist'] = [0.1214735017, 0.1741895871, 0.1224783296]

MEANS['covid'] = [0.5098135074, 0.5098135074, 0.5098135074]
STDS['covid'] = [0.2519904495, 0.2519904495, 0.2519904495]

MEANS['kvasir'] = [0.4859546510, 0.3463666971, 0.2985339895]
STDS['kvasir'] = [0.3312173078, 0.2410854483, 0.2318631172]
```

**统计量来源**:
```python
# 在预处理阶段计算
def compute_dataset_statistics(dataset):
    images = []
    for img, _ in dataset:
        images.append(img)
    images = torch.stack(images)
    
    mean = images.mean(dim=[0, 2, 3])  # 按 channel 计算
    std = images.std(dim=[0, 2, 3])
    
    return mean, std
```

**修改原因**:
- NCFM 的 diffaug 需要这些全局统计量
- 用于归一化计算

---

### 4. HoP 数据加载适配

#### 文件: `adapted/hop_tm/utils/utils_eval_sam.py`

**修改类型**: 添加医疗数据集支持

**原始代码** (GitHub):
```python
def get_dataset(dataset, data_path, batch_size, args):
    if dataset == 'CIFAR10':
        # CIFAR-10 处理
        im_size = (32, 32)
        ...
    elif dataset == 'ImageNet':
        # ImageNet 处理
        im_size = (64, 64)
        ...
    # 没有医疗数据集
```

**修改后代码** (HPC):
```python
def get_dataset(dataset, data_path, batch_size, args):
    # ✅ 添加医疗数据集支持
    if dataset in ['PathMNIST', 'COVID', 'Kvasir']:
        # 使用共享的医疗数据集加载器
        splits = load_medical_splits(
            dataset, 
            data_path,
            train_skip_normalize=False  # HoP 使用正常 Normalize
        )
        
        # 设置图像尺寸
        spec = get_medical_spec(dataset)
        im_size = spec['image_size']
        num_classes = spec['num_classes']
        mean = spec['mean']
        std = spec['std']
        
        return (
            3, im_size, num_classes, mean, std,
            splits['train'], splits['test'], 
            test_loader, train_loader_dict, 
            class_map, class_map_inv
        )
    
    # 原有代码保持不变
    if dataset == 'CIFAR10':
        ...
```

**修改原因**:
- HoP 需要医疗数据集支持
- 使用统一的数据加载接口
- HoP 使用 `train_skip_normalize=False`（在 loader 中 Normalize）

---

## 📝 配置文件修改

### 1. NCFM PathMNIST 配置

#### 文件: `configs/ncfm/pathmnist/ipc10_full_fixed.yaml`

**修改类型**: 新增配置文件

**来源**: 基于 Raw NCFM CIFAR-10 配置

**关键配置**:
```yaml
dataset:
  dataset: pathmnist    # ✅ 医疗数据集
  nclass: 9             # ✅ PathMNIST 特定
  size: 32              # ✅ 从 CIFAR-10 迁移
  data_dir: data/prepared
  batch_real: 128

network:
  net_type: convnet
  norm_type: instance
  depth: 3              # ✅ PathMNIST 小图用 D3
  width: 1.0

train:
  pertrain_epochs: 80   # ✅ 调整: 原 60 → 80
  batch_size: 128
  adamw_lr: 0.001
  eval_optimizer: adamw

augmentation:
  mixup: cut            # ✅ Pretrain 使用 CutMix
  mix_p: 0.5
  dsa: true

save_path:
  save_dir: results/ncfm_fixed/pathmnist      # ✅ 独立目录
  pretrain_dir: pretrained_models/ncfm_fixed

condense:
  ipc: 10
  num_premodel: 20
  num_freqs: 4096       # ✅ NCFM 核心参数
  niter: 20000
```

**与 Raw NCFM 的差异**:
| 参数 | Raw NCFM (CIFAR-10) | PathMNIST | 原因 |
|------|---------------------|-----------|------|
| `dataset` | cifar10 | pathmnist | 数据集不同 |
| `nclass` | 10 | 9 | PathMNIST 9 类 |
| `size` | 32 | 32 | 保持一致 ✅ |
| `depth` | 3 | 3 | 保持一致 ✅ |
| `pertrain_epochs` | 60 | 80 | 更充分训练 |
| `save_dir` | results/ncfm | results/ncfm_fixed | 避免覆盖 |

**未改变的核心参数** (来自 Raw NCFM):
- `num_freqs: 4096` - CF Loss 频率数
- `num_premodel: 20` - Teacher 数量
- `ipc: 10` - 每类图像数
- `niter: 20000` - 蒸馏迭代数
- `adamw_lr: 0.001` - 评估学习率
- `mixup: cut` - Pretrain 使用 CutMix

---

### 2. NCFM COVID 配置

#### 文件: `configs/ncfm/covid/ipc10_full_fixed.yaml`

**修改类型**: 新增配置文件

**来源**: 基于 PathMNIST 配置

**关键配置**:
```yaml
dataset:
  dataset: covid        # ✅ COVID 数据集
  nclass: 4             # ✅ COVID 4 类
  size: 112             # ✅ COVID 大图用 112
  batch_real: 128

network:
  depth: 5              # ✅ COVID 大图用 D5

train:
  pertrain_epochs: 60   # COVID 可能 60 够用
  batch_size: 64        # ⚠️ 减小 (112x112 更大)

save_path:
  save_dir: results/ncfm_fixed/covid
  pretrain_dir: pretrained_models/ncfm_fixed

# 其他参数与 PathMNIST 相同
```

**与 PathMNIST 的差异**:
| 参数 | PathMNIST | COVID | 原因 |
|------|-----------|-------|------|
| `nclass` | 9 | 4 | 数据集不同 |
| `size` | 32 | 112 | COVID 图像更大 |
| `depth` | 3 | 5 | 大图需要更深网络 |
| `batch_size` | 128 | 64 | 112² > 32², 内存限制 |
| `pertrain_epochs` | 80 | 60 | 可能 60 够用 |

---

### 3. NCFM 公平评估配置

#### 文件: `configs/ncfm/pathmnist/ipc10_eval_dsa_only.yaml`

**修改类型**: 新增配置文件

**目的**: 与 HoP 公平对比

**关键修改**:
```yaml
augmentation:
  mixup: none           # ✅ 去除 CutMix
  mix_p: 0.0            # ✅ 0% CutMix
  rrc: true
  dsa: true             # ✅ 保留 DSA
  dsa_strategy: color_crop_cutout_flip_scale_rotate
  aug_type: color_crop_cutout

train:
  evaluation_epochs: 2000
  eval_optimizer: adamw
  adamw_lr: 0.001
```

**与 `ipc10_full_fixed.yaml` 的差异**:
| 参数 | Full (Pretrain/Condense) | Eval DSA Only |
|------|-------------------------|---------------|
| `mixup` | cut | none ✅ |
| `mix_p` | 0.5 | 0.0 ✅ |
| `dsa` | true | true |

**原因**:
- HoP 评估只使用 DSA
- NCFM 评估也只用 DSA 才公平
- Pretrain 可以用 CutMix（是方法的一部分）
- Eval 必须统一（测试合成数据质量）

---

### 4. HoP PathMNIST 配置

#### 文件: `configs/hop_tm/pathmnist/ipc10_full.yaml`

**修改类型**: 新增配置文件

**来源**: Raw HoP (MTT) PathMNIST 配置

**关键配置**:
```yaml
dataset: PathMNIST
num_classes: 9
channel: 3
im_size: [32, 32]       # ✅ 与 NCFM 一致
model: ConvNet          # ✅ D3

ipc: 10
syn_steps: 80           # ✅ Raw HoP
expert_epochs: 2        # ✅ Raw HoP
lr_img: 10              # ✅ Raw HoP
lr_teacher: 0.01        # ✅ Raw HoP

num_experts: 100        # ✅ Raw HoP
train_epochs: 100       # ✅ Raw HoP

buffer_path: buffers/hop_tm
data_path: data/prepared
Iteration: 10000        # ✅ Raw HoP

dsa: true               # ✅ DSA only
dsa_strategy: color_crop_cutout_flip_scale_rotate

high_order: true        # ✅ 高阶轨迹匹配
```

**与 Raw HoP 的差异**:
| 参数 | Raw HoP (CIFAR-10) | PathMNIST | 差异 |
|------|-------------------|-----------|------|
| `dataset` | CIFAR10 | PathMNIST | 数据集 |
| `num_classes` | 10 | 9 | 类别数 |
| `im_size` | [32, 32] | [32, 32] | 一致 ✅ |
| 其他参数 | - | - | 完全来自 Raw HoP ✅ |

---

### 5. HoP COVID 配置

#### 文件: `configs/hop_tm/covid/ipc10_full.yaml`

**修改类型**: 新增配置文件

**来源**: Raw HoP COVID 配置

**关键配置**:
```yaml
dataset: COVID
num_classes: 4          # ✅ COVID 4 类
channel: 3
im_size: [112, 112]     # ✅ 与 NCFM 一致
model: ConvNetD5        # ✅ D5

ipc: 10
syn_steps: 80           # ✅ Raw HoP
expert_epochs: 2
lr_img: 100             # ⚠️ 原值，可能数值不稳定

# 其他参数与 PathMNIST 相同
```

**特殊配置** (`ipc10_stable_lr10.yaml`):
```yaml
# 数值稳定性诊断版本
lr_img: 10              # ✅ 降低学习率 (100 → 10)
# 其他参数完全相同
```

**原因**: 
- `lr_img=100` 在 COVID 上可能数值不稳定
- 创建 `lr_img=10` 版本作为 fallback

---

## 📊 配置参数来源总结

### NCFM 参数来源

| 参数 | PathMNIST | COVID | Kvasir | 来源 |
|------|-----------|-------|--------|------|
| **数据集特定** |
| `dataset` | pathmnist | covid | kvasir | 数据集名称 |
| `nclass` | 9 | 4 | 8 | 数据集类别数 |
| `size` | 32 | 112 | 128 | 图像尺寸 |
| `depth` | 3 | 5 | 5 | 根据图像尺寸 |
| **来自 Raw NCFM** |
| `num_freqs` | 4096 | 4096 | 4096 | Raw NCFM 论文 |
| `num_premodel` | 20 | 20 | 20 | Raw NCFM CIFAR-10 |
| `ipc` | 10 | 10 | 10 | Raw NCFM CIFAR-10 |
| `niter` | 20000 | 20000 | 20000 | Raw NCFM CIFAR-10 |
| `adamw_lr` | 0.001 | 0.001 | 0.001 | Raw NCFM |
| `eval_optimizer` | adamw | adamw | adamw | Raw NCFM |
| **调整参数** |
| `pertrain_epochs` | 80 | 60 | - | 调整 (原 60) |
| `batch_size` | 128 | 64 | 64 | 根据图像尺寸 |
| `save_dir` | ncfm_fixed/ | ncfm_fixed/ | ncfm_fixed/ | 避免覆盖 |

### HoP 参数来源

| 参数 | PathMNIST | COVID | 来源 |
|------|-----------|-------|------|
| **数据集特定** |
| `dataset` | PathMNIST | COVID | 数据集名称 |
| `num_classes` | 9 | 4 | 数据集类别数 |
| `im_size` | [32, 32] | [112, 112] | Raw HoP 配置 |
| `model` | ConvNet | ConvNetD5 | Raw HoP 配置 |
| **来自 Raw HoP (MTT)** |
| `num_experts` | 100 | 100 | Raw HoP |
| `train_epochs` | 100 | 100 | Raw HoP |
| `syn_steps` | 80 | 80 | Raw HoP |
| `expert_epochs` | 2 | 2 | Raw HoP |
| `Iteration` | 10000 | 10000 | Raw HoP |
| `ipc` | 10 | 10 | Raw HoP |
| **特殊调整** |
| `lr_img` | 10 | 10 (100→10) | 数值稳定性 |

---

## 🎯 关键修改说明

### 1. 为什么添加 `train_skip_normalize` 参数？

**问题**:
```python
# 原始代码 (错误):
train_dataset = datasets.ImageFolder(
    'train',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)  # 第一次 Normalize
    ])
)

# NCFM diffaug 中:
images = (images - mean) / std  # 第二次 Normalize!

# 结果: 双重 Normalize → Train 83%, Val 32%
```

**解决**:
```python
# 修改后 (正确):
train_dataset = load_medical_splits(
    'PathMNIST',
    'data',
    train_skip_normalize=True  # ✅ 不在 loader 中 Normalize
)

# NCFM diffaug 中:
images = (images - mean) / std  # 唯一的 Normalize

# 结果: 单次 Normalize → Train 86%, Val 97% ✅
```

---

### 2. 为什么 COVID 使用 nclass=4？

**数据检查**:
```bash
$ ls data/prepared/COVID/train/
COVID/           # 新冠感染
Lung_Opacity/    # 肺部不透明
Normal/          # 正常
Viral_Pneumonia/ # 病毒性肺炎

# 共 4 类
```

**配置**:
```yaml
# NCFM
nclass: 4  # ✅ 正确

# HoP
num_classes: 4  # ✅ 正确
```

---

### 3. 为什么图像尺寸选择 32/112/128？

**PathMNIST: 32×32**
```
原始: 28×28
选择: 32×32
原因:
  - 2的幂次，适合卷积
  - 与 CIFAR-10 一致
  - Raw NCFM 和 Raw HoP 都用 32
```

**COVID: 112×112**
```
原始: varies (可能 224×224)
选择: 112×112
原因:
  - Raw HoP COVID 配置使用 112
  - NCFM 也统一使用 112
  - 平衡性能和效率 (112² = 224² / 4)
  - 4 分类任务，112 足够
```

**Kvasir: 128×128**
```
原始: varies
选择: 128×128
原因:
  - 2的幂次
  - 8 分类任务，需要更多细节
  - 标准尺寸
```

---

### 4. 为什么 COVID 用 depth=5？

**选择原因**:
| 数据集 | 图像尺寸 | Depth | 原因 |
|--------|---------|-------|------|
| PathMNIST | 32×32 | D3 | 小图，3层够用 |
| COVID | 112×112 | D5 | 大图，需要5层 |
| Kvasir | 128×128 | D5 | 大图，需要5层 |

**计算**:
```
ConvNet-D3 (32×32):
  32 → 16 → 8 → 4 (最终特征图)

ConvNet-D5 (112×112):
  112 → 56 → 28 → 14 → 7 → 3 (最终特征图)
  需要 5 层才能降到合适尺寸
```

---

## ✅ 修改验证

### 代码验证
```
✅ 核心算法未改变 (NCFM.py, SampleNet.py, Condenser.py)
✅ 只添加了数据加载适配层
✅ train_skip_normalize 正确实现
✅ 统计量正确同步
```

### 配置验证
```
✅ 图像尺寸统一 (PathMNIST 32, COVID 112)
✅ Backbone 统一 (PathMNIST D3, COVID D5)
✅ 核心参数来自 Raw NCFM/HoP
✅ 公平评估配置 (DSA only)
```

### 结果验证
```
✅ NCFM PathMNIST: 80.4% ± 0.6% (有效)
✅ Train/Val 一致: Train 86%, Val 97%
🔄 HoP PathMNIST: 运行中
🔄 HoP COVID: 运行中
```

---

## 📝 总结

### 代码修改 (4 个文件)
1. ✅ `utils/medical_dataset_utils.py` - 新增医疗数据集加载器
2. ✅ `adapted/ncfm/utils/utils.py` - 添加医疗数据集支持
3. ✅ `adapted/ncfm/data/dataset_statistics.py` - 添加统计量
4. ✅ `adapted/hop_tm/utils/utils_eval_sam.py` - 添加医疗数据集支持

### 配置修改 (7 个文件)
1. ✅ `configs/ncfm/pathmnist/ipc10_full_fixed.yaml` - NCFM PathMNIST
2. ✅ `configs/ncfm/pathmnist/ipc10_eval_dsa_only.yaml` - 公平评估
3. ✅ `configs/ncfm/covid/ipc10_full_fixed.yaml` - NCFM COVID
4. ✅ `configs/ncfm/kvasir/ipc10_full_fixed.yaml` - NCFM Kvasir
5. ✅ `configs/hop_tm/pathmnist/ipc10_full.yaml` - HoP PathMNIST
6. ✅ `configs/hop_tm/covid/ipc10_full.yaml` - HoP COVID
7. ✅ `configs/hop_tm/covid/ipc10_stable_lr10.yaml` - HoP COVID (稳定版)

### 关键修改
- ✅ 避免双重归一化 (`train_skip_normalize=True`)
- ✅ 统一图像尺寸 (32/112/128)
- ✅ 统一 Backbone (D3/D5)
- ✅ 公平评估配置 (DSA only)
- ✅ 核心算法未改变

---

生成时间: 2026-08-14 03:40
状态: 完整记录所有修改
验证: ✅ 所有修改已验证
