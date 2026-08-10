# 配置文件使用指南
# Configuration Usage Guide

## 📁 配置文件结构

```
configs/
├── README.md                          # 配置总览
├── config_loader.py                   # 配置加载工具
├── USAGE_GUIDE.md                     # 本文件
│
├── dc_dsa_dm/                         # DC/DSA/DM配置
│   ├── pathmnist/
│   │   ├── ipc10_dc_quick.yaml       # 快速测试
│   │   └── ipc10_dc_full.yaml        # 完整实验
│   ├── covid/
│   │   └── ipc10_dc_full.yaml
│   └── kvasir/
│       └── ipc10_dc_full.yaml
│
├── hop_tm/                            # HoP-TM配置
│   └── kvasir/
│       └── ipc10_full.yaml
│
├── mtt/                               # MTT配置
│   ├── pathmnist/
│   │   └── ipc10_full.yaml
│   ├── covid/
│   │   └── ipc10_full.yaml
│   └── kvasir/
│       └── ipc10_full.yaml
│
├── datadam/                           # DataDAM配置
│   ├── pathmnist/
│   │   └── ipc10_full.yaml
│   ├── covid/
│   │   └── ipc10_full.yaml
│   └── kvasir/
│       └── ipc10_full.yaml
│
└── cafe/                              # CAFE配置
    ├── pathmnist/
    │   └── ipc10_full.yaml
    ├── covid/
    │   └── ipc10_full.yaml
    └── kvasir/
        └── ipc10_full.yaml
```

## 🚀 快速开始

### 方法1: 使用配置加载器（推荐）

```python
from configs.config_loader import load_config

# 加载配置
config = load_config("configs/dc_dsa_dm/pathmnist/ipc10_dc_full.yaml")

# 获取配置项
dataset = config.get('dataset')
ipc = config.get('ipc')
lr_img = config.get('lr_img')

# 转换为命令行参数
cmd_args = config.to_cmd_args()
# ['--dataset', 'PathMNIST', '--ipc', '10', ...]
```

### 方法2: 直接在命令行引用（如果算法支持）

某些算法可能支持直接加载YAML配置：

```bash
# NCFM直接支持
python condense/condense_script.py --config configs/ncfm/pathmnist.yaml

# HoP-TM可选支持
python distill/distill_high_order_spl.py --config configs/hop_tm/kvasir/ipc10_full.yaml
```

### 方法3: 手动转换为命令行参数

```bash
# 根据配置文件内容，手动构造命令
cd adapted/dc_dsa_dm
python main.py \
  --dataset PathMNIST \
  --method DC \
  --ipc 10 \
  --Iteration 10000 \
  --lr_img 1.0 \
  --batch_train 256 \
  --num_exp 5 \
  --num_eval 5
```

## 📊 各算法配置使用方法

### 1. DC/DSA/DM

**快速测试（验证能否运行）：**
```bash
cd adapted/dc_dsa_dm
python main.py \
  --dataset PathMNIST \
  --method DC \
  --ipc 10 \
  --Iteration 500 \
  --lr_img 1.0 \
  --num_exp 1 \
  --num_eval 3 \
  --epoch_eval_train 50
```

**完整实验（使用配置文件参数）：**
```bash
cd adapted/dc_dsa_dm
python main.py \
  --dataset PathMNIST \
  --method DC \
  --ipc 10 \
  --Iteration 10000 \
  --lr_img 1.0 \
  --batch_train 256 \
  --num_exp 5 \
  --num_eval 5 \
  --epoch_eval_train 300 \
  --dsa \
  --init real
```

**关键参数说明：**
- `--method`: DC/DSA/DM
- `--lr_img`: 合成图像学习率
  - PathMNIST (32x32): 1.0
  - COVID (112x112): 10.0
  - Kvasir (128x128): 100.0
- `--Iteration`: 迭代次数
  - 快速测试: 500
  - 完整实验: 5000-10000

---

### 2. HoP-TM

**使用YAML配置（推荐）：**
```bash
cd adapted/hop_tm
python distill/distill_high_order_spl.py \
  --dataset Kvasir \
  --ipc 10 \
  --Iteration 10000 \
  --lr_img 100 \
  --high_order \
  --base_threshold 5e-9 \
  --growing_factor 1.5 \
  --lamb 0.5
```

**关键参数说明：**
- `--high_order`: 启用高阶匹配（核心功能）
- `--base_threshold`: 基础阈值
- `--growing_factor`: 增长因子
- `--lamb`: 高阶项权重
- `--syn_steps`: 合成步数（80）
- `--expert_epochs`: 专家轨迹epoch数（2）

---

### 3. MTT

**步骤1: 生成Buffer**
```bash
cd adapted/mtt

# PathMNIST
python buffer.py \
  --dataset PathMNIST \
  --model ConvNet \
  --train_epochs 200 \
  --num_experts 100 \
  --batch_train 256

# COVID（小数据集，减少专家数）
python buffer.py \
  --dataset COVID \
  --model ConvNet \
  --train_epochs 150 \
  --num_experts 50 \
  --batch_train 128

# Kvasir（大图像，减小批大小）
python buffer.py \
  --dataset Kvasir \
  --model ConvNet \
  --train_epochs 200 \
  --num_experts 80 \
  --batch_train 64
```

**步骤2: 运行蒸馏**
```bash
# PathMNIST
python distill.py \
  --dataset PathMNIST \
  --ipc 10 \
  --Iteration 10000 \
  --lr_img 1.0 \
  --lr_lr 1e-5 \
  --syn_steps 50 \
  --expert_epochs 3 \
  --buffer_path ./buffers

# COVID
python distill.py \
  --dataset COVID \
  --ipc 10 \
  --Iteration 5000 \
  --lr_img 10.0 \
  --lr_lr 1e-5 \
  --syn_steps 40 \
  --buffer_path ./buffers

# Kvasir
python distill.py \
  --dataset Kvasir \
  --ipc 10 \
  --Iteration 8000 \
  --lr_img 100.0 \
  --lr_lr 1e-5 \
  --syn_steps 50 \
  --buffer_path ./buffers
```

**关键参数说明：**
- `--train_epochs`: Buffer训练轮数
  - 大数据集: 200
  - 小数据集: 150
- `--num_experts`: 专家数量
  - 大数据集: 80-100
  - 小数据集: 50
- `--syn_steps`: 匹配的轨迹长度（40-50）
- `--lr_lr`: 学习率的学习率（MTT特有，1e-5）

---

### 4. NCFM

**使用YAML配置（必须）：**
```bash
cd adapted/ncfm

# PathMNIST
python condense/condense_script.py \
  --config config/ipc10/medical/pathmnist.yaml

# COVID
python condense/condense_script.py \
  --config config/ipc10/medical/covid.yaml

# Kvasir
python condense/condense_script.py \
  --config config/ipc10/medical/kvasir.yaml
```

**配置文件关键参数：**
- `dataset.nclass`: 类别数（9/4/8）
- `dataset.size`: 图像尺寸（32/112/128）
- `network.depth`: 网络深度（3/4/5）
- `condense.num_freqs`: 特征频率数（4096）
- `condense.dis_metrics`: 距离度量（NCFM）
- `train.evaluation_epochs`: 评估轮数（2000）

---

### 5. DataDAM

**命令行运行：**
```bash
cd adapted/datadam

# PathMNIST
python main_DataDAM.py \
  --dataset PathMNIST \
  --ipc 10 \
  --Iteration 10000 \
  --lr_img 1.0 \
  --batch_train 256 \
  --num_exp 5 \
  --num_eval 5

# COVID
python main_DataDAM.py \
  --dataset COVID \
  --ipc 10 \
  --Iteration 5000 \
  --lr_img 10.0 \
  --batch_train 128

# Kvasir
python main_DataDAM.py \
  --dataset Kvasir \
  --ipc 10 \
  --Iteration 8000 \
  --lr_img 100.0 \
  --batch_train 64
```

**关键参数说明：**
- DataDAM特有参数（需要查看原始代码确认）：
  - 注意力图匹配相关参数
  - 可能包括 `--attn_type`, `--match_type` 等

---

### 6. CAFE

**命令行运行：**
```bash
cd adapted/cafe

# PathMNIST
python distill.py \
  --dataset PathMNIST \
  --ipc 10 \
  --Iteration 10000 \
  --lr_img 1.0 \
  --batch_train 256 \
  --num_exp 5 \
  --num_eval 5

# COVID
python distill.py \
  --dataset COVID \
  --ipc 10 \
  --Iteration 5000 \
  --lr_img 10.0 \
  --batch_train 128

# Kvasir
python distill.py \
  --dataset Kvasir \
  --ipc 10 \
  --Iteration 8000 \
  --lr_img 100.0 \
  --batch_train 64
```

**关键参数说明：**
- CAFE特有参数（需要查看原始代码确认）：
  - `--alpha`: 特征对齐权重（0.5）
  - `--beta`: 梯度匹配权重（0.5）

---

## 🎯 推荐参数总结

### 按数据集分类

| 数据集 | 尺寸 | lr_img | batch_train | Iteration | 特点 |
|--------|------|--------|-------------|-----------|------|
| **PathMNIST** | 32×32 | 1.0 | 256 | 10000 | 小尺寸，大数据集 |
| **COVID** | 112×112 | 10.0 | 128 | 5000 | 中尺寸，小数据集 |
| **Kvasir** | 128×128 | 100.0 | 64 | 8000 | 大尺寸，中数据集 |

### 按测试类型分类

| 测试类型 | Iteration | num_exp | num_eval | epoch_eval_train | 用途 |
|----------|-----------|---------|----------|------------------|------|
| **快速验证** | 500 | 1 | 3 | 50 | 验证能否运行 |
| **初步实验** | 1000-3000 | 3 | 3 | 100 | 初步测试效果 |
| **完整实验** | 5000-10000 | 5 | 5 | 300 | 论文实验 |

---

## 🔧 高级用法

### 1. 批量运行实验

创建脚本批量运行不同配置：

```bash
#!/bin/bash
# run_all_experiments.sh

DATASETS=("PathMNIST" "COVID" "Kvasir")
IPCS=(1 5 10 50)
ALGORITHMS=("DC" "DSA" "DM")

for dataset in "${DATASETS[@]}"; do
  for ipc in "${IPCS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
      echo "Running $algo on $dataset with IPC=$ipc"
      python main.py \
        --dataset $dataset \
        --method $algo \
        --ipc $ipc \
        --Iteration 10000
    done
  done
done
```

### 2. 使用Python配置加载器

```python
# run_experiment.py
from configs.config_loader import load_config
import subprocess

# 加载配置
config = load_config("configs/dc_dsa_dm/pathmnist/ipc10_dc_full.yaml")

# 转换为命令行参数
cmd = ["python", "main.py"] + config.to_cmd_args()

# 运行
subprocess.run(cmd, cwd="adapted/dc_dsa_dm")
```

### 3. 配置文件合并

```python
from configs.config_loader import load_config

# 加载基础配置
base_config = load_config("configs/dc_dsa_dm/pathmnist/ipc10_dc_full.yaml")

# 修改特定参数
base_config.update('ipc', 50)
base_config.update('Iteration', 20000)

# 保存为新配置
base_config.save("configs/dc_dsa_dm/pathmnist/ipc50_dc_full.yaml")
```

---

## 📝 配置文件模板

### 创建新配置的步骤

1. 复制现有配置文件
2. 修改数据集相关参数（num_classes, im_size）
3. 调整学习率和批大小
4. 调整迭代次数
5. 根据需要修改算法特定参数

### 示例：创建新的IPC50配置

```yaml
# pathmnist_ipc50_dc_full.yaml
dataset: PathMNIST
method: DC
ipc: 50                    # 改为50
Iteration: 15000           # 增加迭代次数
lr_img: 0.5                # 减小学习率（更多图像）
batch_train: 256
# ... 其他参数
```

---

## ⚠️ 常见问题

### 1. GPU内存不足

**症状**: CUDA out of memory

**解决方案**:
- 减小 `batch_train` 和 `batch_real`
- PathMNIST: 256 → 128
- COVID: 128 → 64
- Kvasir: 64 → 32

### 2. 训练不收敛

**症状**: 评估准确率很低或不稳定

**解决方案**:
- 减小 `lr_img` (除以2或5)
- 增加 `Iteration`
- 确保数据增强 `dsa: True`
- 检查数据路径是否正确

### 3. Buffer生成太慢（MTT）

**症状**: buffer.py运行很久

**解决方案**:
- 减少 `num_experts` (100 → 50)
- 减少 `train_epochs` (200 → 100)
- 快速测试时使用更小的参数

### 4. 配置文件不生效

**症状**: 参数没有被正确加载

**解决方案**:
- 检查YAML语法（缩进、冒号）
- 确认算法支持配置文件加载
- 使用 `config_loader.py` 验证配置

---

## 📚 参考资料

1. **DC/DSA/DM论文**: 查看原始论文获取推荐超参数
2. **HoP-TM论文**: MICCAI 2025，医疗图像专用配置
3. **MTT论文**: CVPR 2022，Buffer生成策略
4. **NCFM论文**: CVPR 2025，YAML配置说明
5. **DataDAM论文**: ICCV 2023，注意力匹配参数
6. **CAFE论文**: CVPR 2022，特征对齐权重

---

## 🎓 最佳实践

1. **先快速测试，再完整实验**
   - 使用 `_quick.yaml` 验证代码
   - 使用 `_full.yaml` 运行正式实验

2. **记录实验配置**
   - 保存每次实验使用的配置文件
   - 在结果目录中复制配置文件

3. **参数调优策略**
   - 固定其他参数，单独调整学习率
   - 固定学习率，调整迭代次数
   - 最后调整批大小和数据增强

4. **GPU资源管理**
   - 大图像（Kvasir）使用小批大小
   - 并行运行多个小数据集实验
   - 使用不同GPU运行不同算法

5. **结果可复现**
   - 固定随机种子 `seed: 0`
   - 保存完整的配置和日志
   - 记录CUDA版本和PyTorch版本
