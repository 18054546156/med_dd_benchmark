# DataDAM 医疗数据集适配版本

## 修改说明

**适配日期**: 2024-08-10  
**原始代码**: `raw/DataDAM/`  
**适配版本**: `adapted/datadam/`

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

DataDAM通过匹配空间注意力图进行数据集蒸馏：

1. **提取多层特征**
2. **计算空间注意力图**
3. **匹配真实数据和合成数据的注意力模式**
4. **更新合成数据**

优势：比梯度匹配更高效，计算成本更低。

## 使用方法

### 运行示例

```bash
# PathMNIST
python main_DataDAM.py --dataset PathMNIST --model ConvNet --ipc 10 \
    --data_path ../data

# COVID
python main_DataDAM.py --dataset COVID --model ConvNet --ipc 10 \
    --data_path ../data

# Kvasir
python main_DataDAM.py --dataset Kvasir --model ConvNet --ipc 10 \
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
- `--attention_param`: 注意力计算参数
  - 0: sum(abs(features))
  - 1: sum(abs(features)^exp)
  - 2: max(abs(features)^exp)

## 输出结果

在指定的结果目录生成：
- 合成数据集
- 注意力图可视化
- 评估结果

## 引用

```bibtex
@inproceedings{datadam2023,
  title={Dataset Distillation via Attention Matching},
  author={Author Names},
  booktitle={ICCV},
  year={2023}
}
```
