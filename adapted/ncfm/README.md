# NCFM 医疗数据集适配版本

## 修改说明

**适配日期**: 2024-08-10  
**原始代码**: `raw/NCFM/`  
**适配版本**: `adapted/ncfm/`

## 修改内容

### 主要修改：添加YAML配置文件

NCFM使用YAML配置文件管理所有参数。为三个医疗数据集创建了专门的配置文件：

**配置文件位置**: `config/ipc10/medical/`

1. **pathmnist.yaml** - PathMNIST病理组织分类
   - 类别数: 9
   - 图像尺寸: 32×32
   - 网络深度: 3

2. **covid.yaml** - COVID-19 X光片分类
   - 类别数: 4
   - 图像尺寸: 112×112
   - 网络深度: 4

3. **kvasir.yaml** - Kvasir消化道内窥镜分类
   - 类别数: 8
   - 图像尺寸: 128×128
   - 网络深度: 5

## 使用方法

### 运行示例

```bash
# PathMNIST
python condense/condense_script.py --config config/ipc10/medical/pathmnist.yaml

# COVID
python condense/condense_script.py --config config/ipc10/medical/covid.yaml

# Kvasir
python condense/condense_script.py --config config/ipc10/medical/kvasir.yaml
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
    └── test/
```

## 配置文件说明

NCFM的配置文件包含以下部分：

### dataset
- `dataset`: 数据集名称
- `nclass`: 类别数
- `size`: 图像尺寸
- `data_dir`: 数据根目录
- `load_memory`: 是否加载到内存（大数据集建议false）

### network
- `net_type`: 网络类型（convnet, resnet等）
- `depth`: 网络深度
- `width`: 网络宽度缩放因子

### condense
- `ipc`: 每类合成图像数量
- `niter`: 训练迭代次数
- `dis_metrics`: 距离度量（NCFM特有）
- `num_freqs`: 频率数量

### augmentation
- `dsa`: 是否使用可微增强
- `dsa_strategy`: 增强策略
- `mixup`: 混合增强类型

## 修改配置

可以根据需要修改YAML文件中的参数：

```yaml
# 修改IPC
condense:
  ipc: 50  # 改为每类50张

# 修改batch size
train:
  batch_size: 32  # 减小batch size节省显存

# 修改保存路径
save_path:
  save_dir: "./my_results"
```

## 引用

```bibtex
@inproceedings{ncfm2025,
  title={Neural Characteristic Function Matching for Dataset Condensation},
  author={Author Names},
  booktitle={CVPR},
  year={2025}
}
```
