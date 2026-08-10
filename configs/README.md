# 医疗数据集蒸馏配置文件说明
# Medical Dataset Distillation Configuration Guide

本目录包含所有6个数据集蒸馏算法在3个医疗数据集上的详细配置文件。

## 目录结构

```
configs/
├── dc_dsa_dm/          # DC/DSA/DM 算法配置
│   ├── pathmnist/
│   ├── covid/
│   └── kvasir/
├── hop_tm/             # HoP-TM 算法配置
│   ├── pathmnist/
│   ├── covid/
│   └── kvasir/
├── mtt/                # MTT 算法配置
│   ├── pathmnist/
│   ├── covid/
│   └── kvasir/
├── ncfm/               # NCFM 算法配置（已在adapted/ncfm/config/中）
├── datadam/            # DataDAM 算法配置
│   ├── pathmnist/
│   ├── covid/
│   └── kvasir/
└── cafe/               # CAFE 算法配置
    ├── pathmnist/
    ├── covid/
    └── kvasir/
```

## 配置文件命名规则

- `ipc{N}_quick.yaml`: 快速测试配置（少量迭代，用于验证）
- `ipc{N}_full.yaml`: 完整实验配置（论文中的参数）
- `ipc{N}_best.yaml`: 最优配置（调优后的最佳参数）

其中 N 通常为: 1, 5, 10, 50

## 数据集特性

| 数据集 | 类别数 | 图像尺寸 | 训练集大小 | 测试集大小 | 特点 |
|--------|--------|----------|-----------|-----------|------|
| PathMNIST | 9 | 32×32 | ~89,996 | ~7,180 | 小尺寸，类别均衡 |
| COVID | 4 | 112×112 | ~2,000 | ~500 | 中尺寸，类别不均衡 |
| Kvasir | 8 | 128×128 | ~6,400 | ~1,600 | 大尺寸，类别不均衡 |

## 关键超参数说明

### 通用参数
- `ipc`: Images Per Class（每类图像数）
- `Iteration`: 蒸馏迭代次数
- `lr_img`: 合成图像学习率
- `batch_train`: 训练批大小
- `num_eval`: 评估次数
- `epoch_eval_train`: 评估时训练轮数

### 数据集特定调整

#### PathMNIST (小尺寸，大数据集)
- 较小的学习率（lr_img: 1-10）
- 较多的迭代次数（Iteration: 10000-20000）
- 较大的批大小（batch_train: 256）

#### COVID (中尺寸，小数据集)
- 中等学习率（lr_img: 10-100）
- 中等迭代次数（Iteration: 5000-10000）
- 中等批大小（batch_train: 128）
- 需要数据增强防止过拟合

#### Kvasir (大尺寸，中数据集)
- 较大的学习率（lr_img: 100-1000）
- 较多的迭代次数（Iteration: 10000-15000）
- 中等批大小（batch_train: 64-128）
- 强数据增强

## 算法特定说明

### DC/DSA/DM
- **DC**: 梯度匹配，适合小数据集
- **DSA**: 可微增强，适合中大数据集
- **DM**: 分布匹配，适合类别不均衡数据集

### HoP-TM
- 高阶轨迹匹配，专为医疗图像设计
- 关键参数: `high_order`, `base_threshold`, `growing_factor`

### MTT
- 需要预先生成专家轨迹buffer
- 关键参数: `syn_steps`, `expert_epochs`

### NCFM
- 基于特征频率匹配
- 关键参数: `num_freqs`, `dis_metrics`

### DataDAM
- 注意力图匹配
- 关键参数: `attn_type`, `match_type`

### CAFE
- 特征对齐 + 梯度匹配
- 关键参数: `alpha`, `beta`（特征/梯度权重）

## 使用方法

### 方法1: 命令行参数（DC/DSA/DM, HoP-TM, MTT, DataDAM, CAFE）
```bash
python main.py --dataset PathMNIST --ipc 10 --lr_img 10 --Iteration 10000
```

### 方法2: YAML配置文件（NCFM, HoP-TM可选）
```bash
python condense_script.py --config configs/ncfm/pathmnist/ipc10_full.yaml
```

### 方法3: 混合模式
```bash
python main.py --config configs/dc/pathmnist/ipc10_full.yaml --ipc 50  # 命令行覆盖配置
```

## 推荐的实验流程

1. **快速验证** (5-10分钟)
   - 使用 `ipc10_quick.yaml`
   - Iteration: 100-500
   - num_eval: 1-3

2. **初步实验** (30-60分钟)
   - 使用 `ipc10_full.yaml`
   - Iteration: 1000-3000
   - num_eval: 3-5

3. **完整实验** (2-8小时)
   - 使用 `ipc10_full.yaml` 的完整参数
   - Iteration: 10000-20000
   - num_eval: 5

4. **最优配置** (调优后)
   - 使用 `ipc10_best.yaml`
   - 基于初步实验结果调整的参数

## 参考文献

各算法的原始论文和推荐配置：

1. **DC**: ICML 2021 - Dataset Condensation with Gradient Matching
2. **DSA**: ICML 2021 - Dataset Condensation with Differentiable Siamese Augmentation
3. **DM**: WACV 2023 - Dataset Condensation with Distribution Matching
4. **MTT**: CVPR 2022 - Dataset Distillation by Matching Training Trajectories
5. **HoP-TM**: MICCAI 2025 - High-Order Progressive Trajectory Matching for Medical Images
6. **NCFM**: CVPR 2025 - Neural Characteristic Function Matching
7. **DataDAM**: ICCV 2023 - DataDAM: Efficient Dataset Distillation with Attention Matching
8. **CAFE**: CVPR 2022 - CAFE: Learning to Condense Dataset by Aligning Features
