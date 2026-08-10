# CAFE 医疗数据集适配版本

## 修改说明

**适配日期**: 2024-08-10  
**原始代码**: `raw/CAFE/`  
**适配版本**: `adapted/cafe/`

## 修改内容

### 主要修改文件：`utils.py`

添加了三个医疗数据集的支持：

1. **PathMNIST** - 病理组织分类
   - 类别数: 9
   - 图像尺寸: 32×32
   - 通道数: 3 (RGB)

2. **COVID** - COVID-19 X光片分类
   - 类别数: 4
   - 图像尺寸: 112×112
   - 通道数: 3 (RGB)

3. **Kvasir** - 消化道内窥镜分类
   - 类别数: 8
   - 图像尺寸: 128×128
   - 通道数: 3 (RGB)

## 核心方法

CAFE (Contrastive Augmentation and Feature Embedding) 通过联合优化进行数据集蒸馏：

1. **特征对齐**: 匹配多层特征分布
2. **梯度匹配**: 同时匹配梯度信息
3. **联合优化**: loss = loss_feature + loss_gradient

优势：收敛比纯梯度匹配快，效果更好。

## 使用方法

### 运行示例

```bash
# PathMNIST
python distill.py --dataset PathMNIST --model ConvNet --ipc 10 \
    --data_path ../data

# COVID
python distill.py --dataset COVID --model ConvNet --ipc 10 \
    --data_path ../data

# Kvasir
python distill.py --dataset Kvasir --model ConvNet --ipc 10 \
    --data_path ../data
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

## 关键参数

- `--dataset`: 数据集名称 (PathMNIST/COVID/Kvasir)
- `--model`: 网络模型 (ConvNet, ResNet18等)
- `--ipc`: 每类合成图像数量
- `--data_path`: 数据根目录
- `--Iteration`: 训练迭代次数
- `--lr_img`: 合成图像学习率
- `--lr_net`: 网络学习率
- `--eval_mode`: 评估模式
  - S: 同一架构
  - M: 多种架构

## 输出结果

在指定的结果目录生成：
- 合成数据集
- 特征可视化
- 评估结果

## 引用

```bibtex
@inproceedings{cafe2022,
  title={CAFE: Learning to Condense Dataset by Aligning Features},
  author={Wang, Kai and others},
  booktitle={CVPR},
  year={2022}
}
```
