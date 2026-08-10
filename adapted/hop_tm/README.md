# HoP-TM 医疗数据集适配版本

## 修改说明

**适配日期**: 2024-08-10  
**原始代码**: `raw/HoP-TM/`  
**适配版本**: `adapted/hop_tm/`

## 修改内容

### 主要修改文件：`utils/utils_baseline.py`

**原始支持**:
- ✅ PathMNIST (已支持)
- ✅ COVID (已支持)

**新增支持**:
- ✅ Kvasir (新添加)

### Kvasir数据集配置

- **类别数**: 8
- **图像尺寸**: 128×128
- **通道数**: 3 (RGB)
- **Mean**: [0.485, 0.456, 0.406] (ImageNet标准)
- **Std**: [0.229, 0.224, 0.225] (ImageNet标准)
- **数据格式**: ImageFolder

## 使用方法

### 运行示例

```bash
# PathMNIST (原本就支持)
python distill/distill_high_order_spl.py --dataset PathMnist --ipc 10

# COVID (原本就支持)
python distill/distill_high_order_spl.py --dataset COVID --ipc 10

# Kvasir (新添加)
python distill/distill_high_order_spl.py --dataset Kvasir --ipc 10
```

### 数据准备

确保数据目录结构正确：

```
data/
├── PathMNIST/     # medmnist自动下载
├── COVID/
│   ├── train/
│   └── test/
└── Kvasir/
    ├── train/
    │   ├── class_0/
    │   ├── class_1/
    │   └── ... (共8个类别)
    └── test/
        └── (同上)
```

## 特点

HoP-TM是专门为医学图像设计的数据集蒸馏方法，包含：

1. **高阶特征匹配**: 不只匹配参数，还匹配梯度和Hessian
2. **渐进式难度调整**: 从简单样本到困难样本逐步匹配
3. **GSAM优化器**: 更稳定的训练过程

## 引用

```bibtex
@inproceedings{bian2025hoptm,
  title={High-Order Progressive Trajectory Matching for Medical Image Dataset Distillation},
  author={Bian, Jinhao and others},
  booktitle={MICCAI},
  year={2025}
}
```
