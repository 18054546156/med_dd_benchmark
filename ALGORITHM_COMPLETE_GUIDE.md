# 医疗数据集蒸馏算法完整指南

## 系统流程图

```
configs/<algorithm>/<dataset>/*.yaml
        ↓
scripts/run_config.py
        ↓
adapted/<algorithm>/ 原始入口
        ↓
data/prepared/<dataset>/train + test
        ↓
results/、buffers/、pretrained_models/、logs/
```

**注意**: `raw/` 目录不参与统一运行，也没有被修改。

---

## 📁 目录结构

```
med_dd_benchmark/
├── adapted/              # 6个已适配的DD算法
│   ├── cafe/
│   ├── datadam/
│   ├── dc_dsa_dm/
│   ├── hop_tm/
│   ├── mtt/
│   └── ncfm/
├── configs/              # 算法配置文件
│   ├── <algorithm>/<dataset>/*.yaml
│   ├── backbones/        # 网络配置参考（不自动加载）
│   └── config_loader.py  # 配置工具（不被run_config.py调用）
├── scripts/
│   └── run_config.py     # 核心运行器
├── utils/
│   └── medical_dataset_utils.py  # 统一数据加载
├── data/prepared/        # 统一数据输入
├── raw/                  # 原始数据（不参与统一运行）
├── buffers/              # Expert trajectories (MTT/HoP-TM生成)
├── results/              # 蒸馏后的合成数据集
├── pretrained_models/    # NCFM预训练模型
└── logs/                 # 实验日志
```

---

## 🎯 核心执行器：`scripts/run_config.py`

### 作用
统一的配置解析和算法调度入口

### 核心函数
- `load_config()` - 加载YAML配置（自己实现，不依赖config_loader.py）
- `command_for()` - 根据算法和阶段生成执行命令
- `require_stage()` - 验证算法支持的阶段
- `resource_path()` - 解析buffer/checkpoint路径
- `resolve_ncfm_load_path()` - NCFM自动选择最终迭代文件（data_{最大数字}.pt，而非data_init.pt）
- `_run_dir()` - 生成时间戳日志目录
- `_write_run_manifest()` - 记录运行元数据（Git commit、data sha256、返回码）

### 运行日志结构
```
logs/<algorithm>/<dataset>/<stage>/<timestamp>/
├── config.yaml          # 配置快照
├── command.txt          # 执行命令
├── run_manifest.json    # 运行元数据
└── stdout.log           # 标准输出
```

---

## 📊 统一数据输入

### 数据结构
所有算法读取相同的数据结构：
```
data/prepared/
├── PathMNIST/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── manifest.json
├── COVID/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── manifest.json
└── Kvasir/
    ├── train/
    ├── val/
    ├── test/
    └── manifest.json
```

### 数据集规格
- **PathMNIST**: 3通道, 32×32, 9类
- **COVID**: 3通道, 112×112, 4类
- **Kvasir**: 3通道, 128×128, 8类

### 统一数据加载接口
`utils/medical_dataset_utils.py` 提供：
- `load_medical_splits()` - 加载train/val/test
- `MedMNISTWrapper()` - 数据集包装器
- `scalarize_label()` - 标签标量化
- `MEDICAL_DATASET_SPECS` - 数据集规格

### 各算法适配器
每个算法有自己的utils适配器，共同调用统一接口：
- `adapted/dc_dsa_dm/utils.py`
- `adapted/datadam/utils.py`
- `adapted/cafe/utils.py`
- `adapted/mtt/utils.py`
- `adapted/hop_tm/utils/utils_gsam.py`
- `adapted/hop_tm/utils/utils_baseline.py`
- `adapted/ncfm/utils/utils.py`

### 数据使用约定
- **DC/DSA/DM, DataDAM, CAFE, MTT, HoP-TM**: 使用 `train` + `test`
- **NCFM**: pretrain/condense使用 `train` + `val`，evaluation使用 `test`

---

## 🔧 六大算法详解

### 1. DC/DSA/DM

**重要**: DC、DSA、DM共用适配目录 `adapted/dc_dsa_dm/`，但算法机制和入口不同，不能共用同一个配置。

#### 配置文件
```
configs/dc_dsa_dm/<dataset>/
├── ipc10_dc_full.yaml    # DC方法
├── ipc10_dsa_full.yaml   # DSA方法
└── ipc10_dm_full.yaml    # DM方法
```

#### 核心参数
```yaml
method: DC/DSA/DM
model: ConvNetD5
ipc: 10
Iteration: 10000
lr_img: 0.1
lr_net: 0.01
init: real
dsa_strategy: color_crop_cutout_flip_scale_rotate
```

#### 执行入口
- **DC/DSA**: `adapted/dc_dsa_dm/main.py`
- **DM**: `adapted/dc_dsa_dm/main_DM.py`

#### 主要函数 (`adapted/dc_dsa_dm/utils.py`)
- `get_dataset()` - 加载数据集
- `get_network()` - 创建ConvNet
- `get_loops()` - 获取训练循环
- `match_loss()` - 梯度/特征匹配损失
- `epoch()` - 训练epoch
- `evaluate_synset()` - 评估合成数据

#### 算法流程
```
读取真实数据 (train + test)
  ↓
初始化合成图像
  ↓
创建 ConvNet
  ↓
DC: 匹配真实梯度和合成梯度
DSA: 梯度匹配 + Siamese augmentation
DM: 匹配真实数据和合成数据的特征分布
  ↓
评估合成数据
  ↓
保存结果
```

#### 生成输出
```
results/dc_dsa_dm/<Dataset>/<Method>/ipc10/
├── res_<Method>_<Dataset>_<Model>_10ipc.pt
└── vis_*.png
```

示例：
```
results/dc_dsa_dm/COVID/DC/ipc10/
results/dc_dsa_dm/COVID/DSA/ipc10/
results/dc_dsa_dm/COVID/DM/ipc10/
```

#### 运行命令
```bash
python scripts/run_config.py `
  --config configs/dc_dsa_dm/covid/ipc10_dc_full.yaml `
  --algorithm dc_dsa_dm `
  --run
```

---

### 2. DataDAM

#### 配置文件
```
configs/datadam/<dataset>/ipc10_full.yaml
```

#### 核心参数
```yaml
model: ConvNetD5
ipc: 10
Iteration: 10000
lr_img: 0.1
lr_net: 0.01
task_balance: 0.5         # DataDAM特有：任务平衡权重
zca: false

attention:                # 注意：说明性配置
  extract_attention: true # run_config.py不传递这些字段
                         # 实际逻辑由get_attention()内部决定
```

#### 执行入口
`adapted/datadam/main_DataDAM.py`

#### 主要函数 (`adapted/datadam/utils.py`)
- `get_network()` - 创建网络 (**修复**: 支持CPU fallback)
- `get_dataset()` - 加载数据集
- `get_attention()` - 生成空间注意力图
- `TensorDataset()` - 合成数据集封装
- `epoch()` - 训练epoch
- `evaluate_synset()` - 评估

#### 算法流程
```
读取真实数据 (train + test)
  ↓
初始化合成数据
  ↓
ConvNet 提取多层 feature
  ↓
get_attention() 生成空间注意力图
  ↓
匹配真实数据和合成数据的 attention
  ↓
task_balance 平衡 attention loss 和分类 loss
  ↓
评估并保存
```

#### 生成输出
```
results/datadam/<Dataset>/ipc10/
├── res_DataDAM_<Dataset>_<Model>_10ipc.pt
└── 训练日志或评估信息
```

#### 修复说明
- 原始代码强制`net.cuda()`，现已支持CPU环境 (utils.py)

#### 运行命令
```bash
python scripts/run_config.py `
  --config configs/datadam/covid/ipc10_full.yaml `
  --algorithm datadam `
  --run
```

---

### 3. CAFE

#### 配置文件
```
configs/cafe/<dataset>/ipc10_full.yaml
```

#### 核心参数
```yaml
model: ConvNetD5
ipc: 10
Iteration: 10000
lr_img: 0.1
lr_net: 0.01

# CAFE特有：特征对齐权重
fourth_weight: 0.1
third_weight: 0.1
second_weight: 0.1
first_weight: 0.1
inner_weight: 1.0
lambda_1: 1.0
lambda_2: 1.0

cafe:                     # 注意：说明性配置
  feature_layers: [2, 3]  # run_config.py不传递此字段
                          # 实际特征层由CAFE网络和代码固定逻辑决定
```

#### 执行入口
`adapted/cafe/distill.py`

**注意**: run_config.py强制传入 `--method DC`，但CAFE的特征对齐逻辑仍然在CAFE代码中执行。

#### 主要函数 (`adapted/cafe/utils.py`)
- `get_dataset()` - 加载数据集
- `get_network()` - 创建网络
- `get_eval_pool()` - 创建评估网络池
- `FeatureMatching()` - **CAFE核心**: 多层特征匹配
- `epoch()` - 训练epoch
- `evaluate_synset()` - 评估

#### 算法流程
```
读取真实数据 (train + test)
  ↓
初始化合成数据
  ↓
提取 ConvNet 多层 feature
  ↓
匹配真实/合成 feature 分布
  ↓
使用 first_weight、second_weight、
third_weight、fourth_weight、
inner_weight、lambda_1、lambda_2
  ↓
双层优化
  ↓
评估并保存
```

#### 生成输出
```
results/cafe/<Dataset>/ipc10/
├── res_DC_<Dataset>_<Model>_10ipc.pt  # 注意：显示DC是因为--method DC
└── vis_*.png
```

#### 运行命令
```bash
python scripts/run_config.py `
  --config configs/cafe/covid/ipc10_full.yaml `
  --algorithm cafe `
  --run
```

---

### 4. MTT (Matching Training Trajectories)

**重要**: MTT必须分两阶段运行，且buffer和distill必须使用相同的model。

#### 配置文件
```
configs/mtt/<dataset>/ipc10_full.yaml
```

#### 核心参数
```yaml
model: ConvNetD5          # buffer和distill必须使用相同模型!
ipc: 10

buffer:                   # 阶段1参数
  model: ConvNetD5        # 必须与顶层model一致
  num_experts: 100
  train_epochs: 50
  save_interval: 10
  lr_teacher: 0.01
  batch_train: 256

distillation:             # 阶段2参数
  Iteration: 10000
  lr_img: 0.1
  lr_lr: 1e-05
  expert_epochs: 3
  syn_steps: 50
  max_start_epoch: 25

buffer_path: buffers/mtt

network:                  # 注意：说明性配置
  depth: 5                # 实际网络由model决定（ConvNetD5）
                         # 修改depth后必须重新生成对应buffer
```

#### 阶段1: Buffer生成

**执行入口**: `adapted/mtt/buffer.py`

**主要函数** (`adapted/mtt/utils.py`):
- `get_dataset()` - 加载数据集
- `get_network()` - 创建网络
- `ParamDiffAug()` - 可微分数据增强
- `epoch()` - 训练epoch

**其他模块**:
- `adapted/mtt/reparam_module.py`:
  - `ReparamModule()` - 参数重参数化

**算法流程**:
```
真实训练集 (train)
  ↓
随机初始化多个 teacher ConvNet
  ↓
训练 train_epochs
  ↓
保存每个 epoch 的模型参数轨迹
  ↓
生成 replay buffer
```

**生成输出**:
```
buffers/mtt/
├── PathMNIST_NO_ZCA/ConvNet/
├── COVID_NO_ZCA/ConvNetD5/
└── Kvasir_NO_ZCA/ConvNetD5/
    ├── replay_buffer_0.pt
    ├── replay_buffer_1.pt
    ├── ...
    └── replay_buffer_99.pt
```

**修复说明**: 修复了医疗数据集的NO_ZCA路径结构 (buffer.py)

**运行命令**:
```bash
python scripts/run_config.py `
  --config configs/mtt/covid/ipc10_full.yaml `
  --algorithm mtt `
  --stage buffer `
  --run
```

#### 阶段2: Distillation

**执行入口**: `adapted/mtt/distill.py`

**主要函数**: 同buffer阶段的utils函数，加上：
- `match_loss()` - 轨迹匹配损失

**Buffer检查**: distill前验证buffer存在性 (run_config.py:376-384)

**算法流程**:
```
读取 replay buffer
  ↓
初始化合成图像
  ↓
创建 ReparamModule
  ↓
选择 expert trajectory
  ↓
匹配合成数据训练轨迹和 expert trajectory
  ↓
评估合成数据
```

**生成输出**:
```
results/mtt/<Dataset>/ipc10/<Dataset>/<wandb_run>/
├── images_*.pt
├── labels_*.pt
├── images_best.pt
└── labels_best.pt
```

**运行命令**:
```bash
python scripts/run_config.py `
  --config configs/mtt/covid/ipc10_full.yaml `
  --algorithm mtt `
  --stage distill `
  --run
```

---

### 5. HoP-TM (High-order Projected Trajectory Matching)

**重要**: HoP-TM必须分两阶段运行。当前正式D5 buffer尚未生成，不能声称三个数据集完整跑通。

#### 配置文件
```
configs/hop_tm/<dataset>/
├── ipc10_full.yaml
└── ipc1_smoke.yaml       # Smoke测试配置
```

#### 核心参数
```yaml
model: ConvNetD5
ipc: 10

# Buffer阶段参数
num_experts: 100
train_epochs: 50
save_interval: 10
lr_teacher: 0.01
batch_train: 256

# HoP-TM特有：GSAM优化器参数
rho_max: 2.0
rho_min: 2.0
alpha: 0.4
adaptive: false
mom: 0.9
l2: 0.0005

# Distillation阶段参数
distillation:
  Iteration: 10000
  lr_img: 0.1
  syn_steps: 50

buffer_path: buffers/hop_tm
```

#### 阶段1: Buffer生成

**执行入口**: `adapted/hop_tm/buffer/buffer_FTD.py`

**主要函数** (`adapted/hop_tm/utils/utils_gsam.py`):
- `get_dataset()` - 加载数据集
- `get_network()` - 创建网络
- `epoch()` - 训练epoch

**GSAM优化器** (`adapted/hop_tm/buffer/gsam/`):
- `GSAM()` - Generalized SAM优化器
- Scheduler和WideResNet等辅助模块

**工具函数** (`adapted/hop_tm/buffer/utility/`):
- `initialize()` - 网络初始化
- `cutout()` - Cutout增强

**算法流程**:
```
真实训练集 (train)
  ↓
训练多个 expert
  ↓
使用 GSAM/FTD 训练
  ↓
保存 expert trajectory
```

**生成输出**:
```
buffers/hop_tm/
├── PathMNIST_NO_ZCA/ConvNet/
├── COVID_NO_ZCA/ConvNetD5/
└── Kvasir_NO_ZCA/ConvNetD5/
    ├── replay_buffer_0.pt
    ├── ...
    └── replay_buffer_99.pt
```

**修复说明**: 修复了GPU检测和导入路径问题 (buffer_FTD.py)

**运行命令**:
```bash
python scripts/run_config.py `
  --config configs/hop_tm/covid/ipc10_full.yaml `
  --algorithm hop_tm `
  --stage buffer `
  --run
```

#### 阶段2: Distillation

**执行入口**: `adapted/hop_tm/distill/distill_high_order_spl.py`

**主要函数** (`adapted/hop_tm/utils/utils_baseline.py`):
- `get_dataset()` - 加载数据集
- `get_network()` - 创建网络
- `evaluate_synset()` - 评估合成数据
- `DiffAugment()` - 数据增强

**核心模块**:
- `HighOrderProjection()` - 高阶投影
- `TrajectoryMatching()` - 轨迹匹配

**Buffer检查**: distill前验证buffer存在性 (run_config.py:411-419)

**算法流程**:
```
加载 expert buffer
  ↓
初始化合成图像
  ↓
匹配一阶/高阶轨迹信息
  ↓
progressive matching
  ↓
更新合成图像和 syn_lr
  ↓
保存合成数据
```

**生成输出**:
```
results/hop_tm/<Dataset>/ipc10/<Dataset>/<wandb_run>/
├── images_*.pt
├── labels_*.pt
├── lr_*.pt
├── images_best.pt
├── labels_best.pt
└── lr_best.pt
```

**运行命令**:
```bash
python scripts/run_config.py `
  --config configs/hop_tm/covid/ipc10_full.yaml `
  --algorithm hop_tm `
  --stage distill `
  --run
```

---

### 6. NCFM (Neural Collapse Feature Matching)

**重要**: NCFM必须三阶段依次运行：pretrain → condense → evaluation。

#### 配置文件
```
configs/ncfm/<dataset>/
├── ipc10_full.yaml
└── ipc1_smoke.yaml
```

#### 核心参数
```yaml
dataset:
  dataset: covid
  nclass: 4
  nch: 3
  size: 112

condense:
  ipc: 10
  iteration: 500          # pretrain迭代
  n_data: 10              # condense迭代轮数

pretrain:
  lr: 0.001
  batch_size: 128

val_repeat: 1             # smoke测试时设为1，正式评估设为10

save_path:
  save_dir: results/ncfm/<dataset>
```

#### 当前网络配置
- **PathMNIST**: ConvNet D3
- **COVID**: ConvNet D5
- **Kvasir**: ConvNet D5

#### 阶段1: Pretrain

**执行入口**: `adapted/ncfm/pretrain/pretrain_script.py`

**主要函数** (`adapted/ncfm/utils/utils.py`):
- `define_model()` - 定义模型
- `get_loader()` - 获取数据加载器
- `train_epoch()` - 训练epoch
- `validate()` - 验证

**配置处理** (`adapted/ncfm/argsprocessor/args.py`):
- `args_checker()` - **修复**: UTF-8配置加载，支持中文注释

**初始化** (`adapted/ncfm/utils/init_script.py`):
- `init_device()` - **修复**: Windows动态端口选择，避免DDP冲突

**算法流程**:
```
加载真实 train + val 数据
  ↓
初始化多个模型（seed0-seed19）
  ↓
训练 pretrain 迭代
  ↓
保存初始和训练后模型
```

**生成输出**:
```
pretrained_models/ncfm/
└── <Dataset>/
    └── ConvNetD<depth>_IN_W128/
        └── seed0/
            ├── premodel0_init.pth.tar
            ├── premodel0_trained.pth.tar
            ├── ...
            └── premodel19_trained.pth.tar
```

**运行命令**:
```bash
python scripts/run_config.py `
  --config configs/ncfm/covid/ipc10_full.yaml `
  --algorithm ncfm `
  --stage pretrain `
  --run
```

#### 阶段2: Condense

**执行入口**: `adapted/ncfm/condense/condense_script.py`

**主要函数**:
- `adapted/ncfm/utils/utils.py`:
  - `get_loader()` - 数据加载
  - `get_feature_extractor()` - 获取特征提取器
  - `update_feature_extractor()` - 更新特征提取器
- `adapted/ncfm/condenser/Condenser.py`:
  - `load_condensed_data()` - 加载合成数据
  - `condense()` - 执行蒸馏
- `adapted/ncfm/condenser/compute_loss.py`:
  - `neural_collapse_loss()` - Neural Collapse损失

**算法流程**:
```
加载真实 train + val 数据
  ↓
加载 pretrain 初始/训练后模型
  ↓
创建 model_init、model_interval、model_final
  ↓
计算 Neural Characteristic Function
  ↓
优化合成数据
  ↓
保存不同迭代点的数据
```

**生成输出**:
```
results/ncfm/<dataset>/condense/<dataset>/ipc10/<run_id>/
├── args.log
├── images/
│   └── img_*.png
└── distilled_data/
    ├── data_init.pt
    ├── data_1000.pt
    ├── data_2000.pt
    ├── ...
    └── data_20000.pt     # 最终迭代
```

**运行命令**:
```bash
python scripts/run_config.py `
  --config configs/ncfm/covid/ipc10_full.yaml `
  --algorithm ncfm `
  --stage condense `
  --run
```

#### 阶段3: Evaluation

**执行入口**: `adapted/ncfm/evaluation/evaluation_script.py`

**主要函数** (`adapted/ncfm/condenser/evaluate.py`):
- `evaluate_synset()` - 评估合成数据集

**val_repeat参数** (run_config.py:447):
- smoke测试: `val_repeat: 1` (快速验证)
- 正式评估: `val_repeat: 10` (默认，10次平均)

**load_path解析** (run_config.py:179-232):
- `resolve_ncfm_load_path()` 自动从目录中选择最终迭代文件
- **目录模式**: 优先选择data_{最大数字}.pt（如data_20000.pt），而非data_init.pt
- **文件模式**: 直接使用指定的.pt文件

**算法流程**:
```
加载 data_20000.pt（或指定文件）
  ↓
使用 test split
  ↓
重新训练评估模型
  ↓
计算 accuracy
  ↓
重复 val_repeat 次取平均
```

**生成输出**:
```
results/ncfm/<dataset>/evaluate/<dataset>/ipc10/<run_id>/
├── args.log
└── evaluation log/result
```

**运行命令**:
```bash
python scripts/run_config.py `
  --config configs/ncfm/covid/ipc10_full.yaml `
  --algorithm ncfm `
  --stage evaluation `
  --load-path results/ncfm/covid/condense/covid/ipc10/<run_id>/distilled_data `
  --run
```

---

## 🔄 配置加载机制

### `configs/config_loader.py`

**用途**:
- 手动读取YAML
- 展平嵌套配置
- 检查键冲突
- 生成命令行参数
- 查找配置路径

**注意**: 当前 `scripts/run_config.py` 没有导入它。真正运行时使用的是 `run_config.py` 自己的 `load_config()` 和 `cfg()` 函数。

### `configs/backbones/*.yaml`

**用途**:
- 记录网络配置参考（PathMNIST D3, COVID D5, Kvasir D5）
- 用于 `scripts/validate_config.py` 和 `scripts/check_fair_setup.sh`
- 检查具体算法配置是否一致

**注意**: 
- 不会自动覆盖具体算法YAML
- 不会自动把网络参数传给算法
- 仅作为配置验证的参考标准

---

## 🛡️ 运行时安全检查

### 阶段验证 (`require_stage()`)
- **DC/DSA/DM, DataDAM, CAFE**: 只支持 `distill`/`smoke`
- **MTT, HoP-TM**: 只支持 `buffer`/`distill`
- **NCFM**: 只支持 `pretrain`/`condense`/`evaluation`

### 资源检查
- **MTT distill前**: 检查buffer存在性 (run_config.py:376-384)
- **HoP-TM distill前**: 检查buffer存在性 (run_config.py:411-419)
- **NCFM evaluation前**: 检查load_path有效性 (run_config.py:448-451)

### 配置验证 (`validate_contract()`)
- 验证num_classes/channel/im_size与数据集匹配

### 模型一致性
- **MTT**: buffer.model必须与顶层model一致 (run_config.py:342-346)

---

## 📂 生成文件汇总

| 算法 | 阶段 | 生成位置 | 说明 |
|------|------|----------|------|
| DC/DSA/DM | distill | `results/dc_dsa_dm/<Dataset>/<Method>/ipc10/` | `.pt` 文件 |
| DataDAM | distill | `results/datadam/<Dataset>/ipc10/` | `.pt` 文件 |
| CAFE | distill | `results/cafe/<Dataset>/ipc10/` | `.pt` 文件 |
| MTT | buffer | `buffers/mtt/<Dataset>_NO_ZCA/<Model>/` | `replay_buffer_*.pt` |
| MTT | distill | `results/mtt/<Dataset>/ipc10/<Dataset>/<wandb_run>/` | `images_*.pt`, `labels_*.pt` |
| HoP-TM | buffer | `buffers/hop_tm/<Dataset>_NO_ZCA/<Model>/` | `replay_buffer_*.pt` |
| HoP-TM | distill | `results/hop_tm/<Dataset>/ipc10/<Dataset>/<wandb_run>/` | `images_*.pt`, `labels_*.pt`, `lr_*.pt` |
| NCFM | pretrain | `pretrained_models/ncfm/<Dataset>/ConvNetD<N>_IN_W128/seed0/` | `.pth.tar` 文件 |
| NCFM | condense | `results/ncfm/<dataset>/condense/<dataset>/ipc10/<run_id>/distilled_data/` | `data_*.pt` |
| NCFM | evaluation | `results/ncfm/<dataset>/evaluate/<dataset>/ipc10/<run_id>/` | 日志和结果 |
| 所有 | 所有 | `logs/<algorithm>/<dataset>/<stage>/<timestamp>/` | 运行日志和元数据 |

---

## 🔧 关键修复点总结

1. **DataDAM** (`adapted/datadam/utils.py`):
   - 支持CPU fallback，不强制CUDA

2. **MTT** (`adapted/mtt/buffer.py`):
   - 修复医疗数据集NO_ZCA buffer路径

3. **HoP-TM** (`adapted/hop_tm/buffer/buffer_FTD.py`):
   - 修复GPU检测和导入路径

4. **NCFM** (`adapted/ncfm/argsprocessor/args.py`):
   - UTF-8配置读取，支持中文注释

5. **NCFM** (`adapted/ncfm/utils/init_script.py`):
   - Windows动态DDP端口选择

6. **配置加载** (`configs/config_loader.py`):
   - 配置键冲突检测（虽然未被run_config.py使用）

7. **运行器** (`scripts/run_config.py`):
   - 阶段检查
   - 资源路径解析
   - NCFM路径自动解析（优先data_{最大数字}.pt）
   - 运行manifest记录

8. **Smoke配置** (所有`configs/*/ipc1_smoke.yaml`):
   - NCFM: `val_repeat: 1` 快速验证

---

## 📚 算法对比表

| 算法 | 阶段数 | Buffer | 核心特点 | 适配器 |
|------|--------|--------|----------|--------|
| DC | 1 | ❌ | 基础梯度匹配 | `dc_dsa_dm/utils.py` |
| DSA | 1 | ❌ | 梯度匹配 + Siamese增强 | `dc_dsa_dm/utils.py` |
| DM | 1 | ❌ | 特征分布匹配 | `dc_dsa_dm/utils.py` |
| DataDAM | 1 | ❌ | 注意力匹配 + 任务平衡 | `datadam/utils.py` |
| CAFE | 1 | ❌ | 多层特征对齐 | `cafe/utils.py` |
| MTT | 2 | ✅ | 轨迹匹配 | `mtt/utils.py` |
| HoP-TM | 2 | ✅ | 高阶投影 + GSAM | `hop_tm/utils/utils_gsam.py`, `utils_baseline.py` |
| NCFM | 3 | ❌ | Neural Collapse | `ncfm/utils/utils.py` |

---

## 📖 相关文档

- [CHANGES_ANALYSIS.md](CHANGES_ANALYSIS.md) - 修改分析和必要性说明
- [HOP_NCFM_PIPELINE.md](HOP_NCFM_PIPELINE.md) - HoP-TM和NCFM详细流程
- [CONFIG_PRINCIPLES.md](configs/CONFIG_PRINCIPLES.md) - 配置原则
- [CONVNETD5_SUPPORT.md](CONVNETD5_SUPPORT.md) - ConvNetD5支持文档
- [STATUS_REPORT.md](STATUS_REPORT.md) - 项目状态报告
