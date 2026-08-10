# MTT 医疗数据集适配版本

## 修改说明

**适配日期**: 2024-08-10  
**原始代码**: `raw/mtt-distillation/`  
**适配版本**: `adapted/mtt/`

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

### MTT特殊处理

MTT方法需要按类别分组的数据加载器（`loader_train_dict`），用于轨迹匹配时分别处理每个类别。

## 使用方法

### ⚠️ 重要：需要先生成专家轨迹

MTT方法需要预先在真实数据上训练专家网络，生成参数轨迹buffer文件。

### 步骤1: 生成专家轨迹

```bash
# PathMNIST
python buffer.py --dataset PathMNIST --model ConvNet --train_epochs 50 \
    --num_experts 100 --data_path ../data --buffer_path ../buffers/PathMNIST

# COVID
python buffer.py --dataset COVID --model ConvNet --train_epochs 50 \
    --num_experts 100 --data_path ../data --buffer_path ../buffers/COVID

# Kvasir
python buffer.py --dataset Kvasir --model ConvNet --train_epochs 50 \
    --num_experts 100 --data_path ../data --buffer_path ../buffers/Kvasir
```

**注意**: 
- 每个数据集生成buffer约需5-10分钟
- Buffer文件较大（约1-5GB per dataset）
- 确保有足够的磁盘空间

### 步骤2: 运行蒸馏

```bash
# PathMNIST
python distill.py --dataset PathMNIST --model ConvNet --ipc 10 \
    --buffer_path ../buffers/PathMNIST --data_path ../data

# COVID
python distill.py --dataset COVID --model ConvNet --ipc 10 \
    --buffer_path ../buffers/COVID --data_path ../data

# Kvasir
python distill.py --dataset Kvasir --model ConvNet --ipc 10 \
    --buffer_path ../buffers/Kvasir --data_path ../data
```

## 关键参数

### buffer.py参数
- `--dataset`: 数据集名称
- `--model`: 网络模型（ConvNet, ResNet18等）
- `--train_epochs`: 专家训练轮数（默认50）
- `--num_experts`: 专家数量（默认100）
- `--buffer_path`: buffer保存路径

### distill.py参数
- `--dataset`: 数据集名称
- `--ipc`: 每类合成图像数量
- `--buffer_path`: 专家轨迹路径
- `--syn_steps`: 合成数据训练步数
- `--expert_epochs`: 使用的专家epoch数
- `--max_start_epoch`: 轨迹起始点范围

## 输出结果

在指定的结果目录生成：
- 合成数据集
- 评估结果
- 训练日志

## 引用

```bibtex
@inproceedings{cazenavette2022dataset,
  title={Dataset Distillation by Matching Training Trajectories},
  author={Cazenavette, George and Wang, Tongzhou and Torralba, Antonio and Efros, Alexei A and Isola, Jun},
  booktitle={CVPR},
  year={2022}
}
```
