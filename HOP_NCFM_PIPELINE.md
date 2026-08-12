# HoP-TM 和 NCFM 完整实验流程文档

**日期**: 2026-08-12  
**目的**: 详细说明两个算法在三个数据集上的配置、流程、输入输出

---

## 一、HoP-TM (High-order Trajectory Matching)

### 1.1 配置文件

| 数据集 | 配置文件 | Backbone | IPC |
|--------|---------|----------|-----|
| PathMNIST | `configs/hop_tm/pathmnist/ipc10_full.yaml` | ConvNet (D3) | 10 |
| PathMNIST | `configs/hop_tm/pathmnist/ipc1_smoke.yaml` | ConvNet (D3) | 1 |
| COVID | `configs/hop_tm/covid/ipc10_full.yaml` | ConvNetD5 | 10 |
| COVID | `configs/hop_tm/covid/ipc1_smoke.yaml` | ConvNetD5 | 1 |
| Kvasir | `configs/hop_tm/kvasir/ipc10_full.yaml` | ConvNetD5 | 10 |
| Kvasir | `configs/hop_tm/kvasir/ipc1_smoke.yaml` | ConvNetD5 | 1 |

### 1.2 两阶段流程

#### 阶段 1: Buffer Generation (Expert Trajectories)

**目的**: 预训练多个 expert models，保存训练轨迹供蒸馏使用

**入口脚本**: `adapted/hop_tm/buffer/buffer_FTD.py`

**运行命令**:
```bash
# PathMNIST
python scripts/run_config.py configs/hop_tm/pathmnist/ipc10_full.yaml --stage buffer

# COVID
python scripts/run_config.py configs/hop_tm/covid/ipc10_full.yaml --stage buffer

# Kvasir
python scripts/run_config.py configs/hop_tm/kvasir/ipc10_full.yaml --stage buffer
```

**实际执行命令** (由 run_config.py 生成):
```bash
python adapted/hop_tm/buffer/buffer_FTD.py \
  --dataset PathMNIST \
  --model ConvNet \
  --num_experts 100 \
  --train_epochs 100 \
  --batch_real 256 \
  --lr_teacher 0.01 \
  --dsa true \
  --data_path data/prepared \
  --buffer_path buffers/hop_tm \
  --zca false
```

**输入**:
- 原始训练数据: `data/prepared/PathMNIST/train/`
- 配置: `num_experts=100`, `train_epochs=100`

**输出目录结构**:
```
buffers/hop_tm/
└── PathMNIST/            # 或 COVID, Kvasir
    └── ConvNet/          # 或 ConvNetD5
        ├── expert_0.pth
        ├── expert_1.pth
        ├── ...
        └── expert_99.pth
```

**如果 zca=false**, 目录名为:
```
buffers/hop_tm/
└── PathMNIST_NO_ZCA/     # 加 _NO_ZCA 后缀
    └── ConvNet/
        └── expert_*.pth
```

**每个 expert_*.pth 包含**:
```python
{
    'model_state_dict': ...,  # 模型权重
    'optimizer_state_dict': ...,  # 优化器状态
    'epoch': int,  # 训练轮数
    'trajectories': [...],  # 训练轨迹参数
}
```

**预计耗时**:
- PathMNIST: ~10-20 小时 (100 experts × 100 epochs)
- COVID: ~20-30 小时 (更大的 D5 网络)
- Kvasir: ~20-30 小时

---

#### 阶段 2: Distillation (合成数据生成)

**目的**: 使用 buffer 中的 expert trajectories 匹配合成数据

**入口脚本**: `adapted/hop_tm/distill/distill_high_order_spl.py`

**运行命令**:
```bash
# PathMNIST
python scripts/run_config.py configs/hop_tm/pathmnist/ipc10_full.yaml --stage distill

# COVID
python scripts/run_config.py configs/hop_tm/covid/ipc10_full.yaml --stage distill

# Kvasir
python scripts/run_config.py configs/hop_tm/kvasir/ipc10_full.yaml --stage distill
```

**实际执行命令**:
```bash
python adapted/hop_tm/distill/distill_high_order_spl.py \
  --cfg configs/hop_tm/pathmnist/ipc10_full.yaml \
  --data_path data/prepared \
  --buffer_path buffers/hop_tm \
  --save_path results/hop_tm/PathMNIST/ipc10
```

**前置检查** (run_config.py 自动执行):
```python
# 检查 buffer 是否存在
expected = buffers/hop_tm/PathMNIST/ConvNet/
if not expected.exists():
    raise FileNotFoundError("需要先运行 buffer 阶段")
```

**输入**:
- Expert buffer: `buffers/hop_tm/PathMNIST/ConvNet/expert_*.pth`
- 原始数据: `data/prepared/PathMNIST/` (用于初始化和验证)
- 配置: `Iteration=10000`, `lr_img=10`, `high_order=true`

**输出目录结构**:
```
results/hop_tm/PathMNIST/ipc10/
├── images_best.png           # 最佳合成图像可视化
├── images_last.png           # 最后一轮合成图像
├── res_PathMNIST_ConvNet_10ipc.pt   # 合成数据 tensor
├── checkpoints/
│   ├── ckpt_0500.pt         # 每 save_interval 保存
│   ├── ckpt_1000.pt
│   └── ...
└── logs/
    └── training.log         # 训练日志
```

**res_*.pt 文件格式**:
```python
{
    'data': torch.Tensor,     # shape: [num_classes * ipc, channel, H, W]
                              # PathMNIST: [90, 3, 32, 32] (9类×10IPC)
    'label': torch.Tensor,    # shape: [num_classes * ipc]
    'soft_label': torch.Tensor,  # 软标签（如果使用）
}
```

**预计耗时**:
- PathMNIST: ~5-10 小时 (10000 iterations)
- COVID: ~10-15 小时
- Kvasir: ~10-15 小时

---

### 1.3 核心参数说明

#### PathMNIST 特有参数
```yaml
model: ConvNet           # D3
lr_img: 10               # RAW 作者设置
Iteration: 10000
syn_steps: 80
expert_epochs: 2
high_order: true
```

#### COVID 特有参数
```yaml
model: ConvNetD5         # D5
lr_img: 100              # 比 PathMNIST 大 10 倍
Iteration: 10000
syn_steps: 20
expert_epochs: 3
high_order: true
```

#### Kvasir 特有参数
```yaml
model: ConvNetD5         # D5
lr_img: 100              # 参考 COVID
Iteration: 10000
syn_steps: 20
expert_epochs: 3
high_order: true
```

---

## 二、NCFM (Neural Collapse inspired Feature Matching)

### 2.1 配置文件

| 数据集 | 配置文件 | Backbone | IPC |
|--------|---------|----------|-----|
| PathMNIST | `configs/ncfm/pathmnist/ipc10_full.yaml` | ConvNet (depth=3) | 10 |
| PathMNIST | `configs/ncfm/pathmnist/ipc1_smoke.yaml` | ConvNet (depth=3) | 1 |
| COVID | `configs/ncfm/covid/ipc10_full.yaml` | ConvNet (depth=5) | 10 |
| COVID | `configs/ncfm/covid/ipc1_smoke.yaml` | ConvNet (depth=5) | 1 |
| Kvasir | `configs/ncfm/kvasir/ipc10_full.yaml` | ConvNet (depth=5) | 10 |
| Kvasir | `configs/ncfm/kvasir/ipc1_smoke.yaml` | ConvNet (depth=5) | 1 |

### 2.2 三阶段流程

#### 阶段 1: Pretrain (预训练多个模型)

**目的**: 预训练 20 个模型用于后续 feature matching

**入口脚本**: `adapted/ncfm/main.py --run_mode pretrain`

**运行命令**:
```bash
# PathMNIST
python scripts/run_config.py configs/ncfm/pathmnist/ipc10_full.yaml --stage pretrain

# COVID
python scripts/run_config.py configs/ncfm/covid/ipc10_full.yaml --stage pretrain

# Kvasir
python scripts/run_config.py configs/ncfm/kvasir/ipc10_full.yaml --stage pretrain
```

**实际执行命令**:
```bash
cd adapted/ncfm && python main.py \
  --config ../../configs/ncfm/pathmnist/ipc10_full.yaml \
  --run_mode pretrain \
  --gpu 0
```

**输入**:
- 原始训练数据: `data/prepared/PathMNIST/train/`
- 配置: `model_num=20`, `pertrain_epochs=60`

**输出目录结构**:
```
pretrained_models/ncfm/
└── pathmnist_convnet_d3/    # 或 covid_convnet_d5, kvasir_convnet_d5
    ├── model_0.pth
    ├── model_1.pth
    ├── ...
    └── model_19.pth
```

**每个 model_*.pth 包含**:
```python
{
    'model_state_dict': ...,  # 模型权重
    'accuracy': float,         # 验证集准确率
    'epoch': int,              # 训练轮数
}
```

**预计耗时**:
- PathMNIST: ~5-10 小时 (20 models × 60 epochs)
- COVID: ~10-15 小时
- Kvasir: ~10-15 小时

---

#### 阶段 2: Condense (数据蒸馏)

**目的**: 使用预训练模型进行 feature matching，生成合成数据

**入口脚本**: `adapted/ncfm/main.py --run_mode condense`

**运行命令**:
```bash
# PathMNIST
python scripts/run_config.py configs/ncfm/pathmnist/ipc10_full.yaml --stage condense

# COVID
python scripts/run_config.py configs/ncfm/covid/ipc10_full.yaml --stage condense

# Kvasir
python scripts/run_config.py configs/ncfm/kvasir/ipc10_full.yaml --stage condense
```

**实际执行命令**:
```bash
cd adapted/ncfm && python main.py \
  --config ../../configs/ncfm/pathmnist/ipc10_full.yaml \
  --run_mode condense \
  --gpu 0 \
  --ipc 10
```

**前置检查** (run_config.py 自动执行):
```python
# 检查 pretrain 模型是否存在
expected = pretrained_models/ncfm/pathmnist_convnet_d3/
if not expected.exists() or len(list(expected.glob("model_*.pth"))) < 20:
    raise FileNotFoundError("需要先运行 pretrain 阶段")
```

**输入**:
- 预训练模型: `pretrained_models/ncfm/pathmnist_convnet_d3/model_*.pth`
- 原始数据: `data/prepared/PathMNIST/`
- 配置: `niter=20000`, `num_freqs=4096`, `dis_metrics=NCFM`

**输出目录结构**:
```
results/ncfm/pathmnist/
└── distilled_data/
    ├── data_init.pt         # 初始化的合成数据
    ├── data_1000.pt         # iter=1000 的合成数据
    ├── data_2000.pt
    ├── ...
    └── data_20000.pt        # 最终合成数据
```

**data_*.pt 文件格式**:
```python
{
    'data': torch.Tensor,     # shape: [num_classes * ipc, channel, H, W]
                              # PathMNIST: [90, 3, 32, 32]
    'label': torch.Tensor,    # shape: [num_classes * ipc]
    'iteration': int,         # 当前迭代次数
}
```

**预计耗时**:
- PathMNIST: ~10-15 小时 (20000 iterations)
- COVID: ~15-20 小时
- Kvasir: ~15-20 小时

---

#### 阶段 3: Evaluation (评估)

**目的**: 在合成数据上训练模型，在真实测试集上评估

**入口脚本**: `adapted/ncfm/main.py --run_mode evaluation`

**运行命令**:
```bash
# PathMNIST - 评估最终合成数据
python scripts/run_config.py configs/ncfm/pathmnist/ipc10_full.yaml \
  --stage evaluation \
  --load-path results/ncfm/pathmnist/distilled_data

# 或者指定具体文件
python scripts/run_config.py configs/ncfm/pathmnist/ipc10_full.yaml \
  --stage evaluation \
  --load-path results/ncfm/pathmnist/distilled_data/data_20000.pt
```

**实际执行命令**:
```bash
cd adapted/ncfm && python main.py \
  --config ../../configs/ncfm/pathmnist/ipc10_full.yaml \
  --run_mode evaluation \
  --gpu 0 \
  --ipc 10 \
  --val_repeat 10 \
  --load_path ../../results/ncfm/pathmnist/distilled_data/data_20000.pt
```

**路径自动解析** (run_config.py 功能):
- 如果传目录: 自动选择 `data_init.pt` 或最新的 `data_*.pt`
- 如果传文件: 直接使用

**输入**:
- 合成数据: `results/ncfm/pathmnist/distilled_data/data_20000.pt`
- 测试数据: `data/prepared/PathMNIST/test/`
- 配置: `evaluation_epochs=2000`, `val_repeat=10`

**输出目录结构**:
```
results/ncfm/pathmnist/
└── evaluation/
    ├── eval_results.json    # 评估结果
    │   {
    │     "mean_accuracy": 0.XX,
    │     "std_accuracy": 0.XX,
    │     "accuracies": [0.XX, 0.XX, ...],  # 10 次评估
    │     "load_path": "...",
    │   }
    └── logs/
        └── eval.log
```

**预计耗时**:
- PathMNIST: ~5 小时 (10 runs × 2000 epochs)
- COVID: ~8 小时
- Kvasir: ~8 小时

---

### 2.3 核心参数说明

#### PathMNIST 特有参数
```yaml
network:
  depth: 3               # D3
  width: 1.0
dataset:
  size: 32
  nclass: 9
condense:
  ipc: 10
  niter: 20000
  num_freqs: 4096
  num_premodel: 20
train:
  pertrain_epochs: 60
  evaluation_epochs: 2000
```

#### COVID 特有参数
```yaml
network:
  depth: 5               # D5
  width: 1.0
dataset:
  size: 112
  nclass: 4
condense:
  ipc: 10
  niter: 20000
  num_freqs: 4096
  num_premodel: 20
train:
  pertrain_epochs: 60
  evaluation_epochs: 2000
```

#### Kvasir 特有参数
```yaml
network:
  depth: 5               # D5
  width: 1.0
dataset:
  size: 128
  nclass: 8
condense:
  ipc: 10
  niter: 20000
  num_freqs: 4096
  num_premodel: 20
train:
  pertrain_epochs: 60
  evaluation_epochs: 2000
```

---

## 三、完整实验流程时间线

### HoP-TM 完整流程

#### PathMNIST
```bash
# 第 1 天: Buffer (10-20h)
python scripts/run_config.py configs/hop_tm/pathmnist/ipc10_full.yaml --stage buffer

# 第 2 天: Distill (5-10h)
python scripts/run_config.py configs/hop_tm/pathmnist/ipc10_full.yaml --stage distill

# 输出: results/hop_tm/PathMNIST/ipc10/res_PathMNIST_ConvNet_10ipc.pt
```

#### COVID
```bash
# 第 1-2 天: Buffer (20-30h)
python scripts/run_config.py configs/hop_tm/covid/ipc10_full.yaml --stage buffer

# 第 3 天: Distill (10-15h)
python scripts/run_config.py configs/hop_tm/covid/ipc10_full.yaml --stage distill

# 输出: results/hop_tm/COVID/ipc10/res_COVID_ConvNetD5_10ipc.pt
```

#### Kvasir
```bash
# 第 1-2 天: Buffer (20-30h)
python scripts/run_config.py configs/hop_tm/kvasir/ipc10_full.yaml --stage buffer

# 第 3 天: Distill (10-15h)
python scripts/run_config.py configs/hop_tm/kvasir/ipc10_full.yaml --stage distill

# 输出: results/hop_tm/Kvasir/ipc10/res_Kvasir_ConvNetD5_10ipc.pt
```

---

### NCFM 完整流程

#### PathMNIST
```bash
# 第 1 天: Pretrain (5-10h)
python scripts/run_config.py configs/ncfm/pathmnist/ipc10_full.yaml --stage pretrain

# 第 2 天: Condense (10-15h)
python scripts/run_config.py configs/ncfm/pathmnist/ipc10_full.yaml --stage condense

# 第 3 天: Evaluation (5h)
python scripts/run_config.py configs/ncfm/pathmnist/ipc10_full.yaml \
  --stage evaluation \
  --load-path results/ncfm/pathmnist/distilled_data

# 输出: results/ncfm/pathmnist/distilled_data/data_20000.pt
```

#### COVID
```bash
# 第 1 天: Pretrain (10-15h)
python scripts/run_config.py configs/ncfm/covid/ipc10_full.yaml --stage pretrain

# 第 2 天: Condense (15-20h)
python scripts/run_config.py configs/ncfm/covid/ipc10_full.yaml --stage condense

# 第 3 天: Evaluation (8h)
python scripts/run_config.py configs/ncfm/covid/ipc10_full.yaml \
  --stage evaluation \
  --load-path results/ncfm/covid/distilled_data

# 输出: results/ncfm/covid/distilled_data/data_20000.pt
```

#### Kvasir
```bash
# 第 1 天: Pretrain (10-15h)
python scripts/run_config.py configs/ncfm/kvasir/ipc10_full.yaml --stage pretrain

# 第 2 天: Condense (15-20h)
python scripts/run_config.py configs/ncfm/kvasir/ipc10_full.yaml --stage condense

# 第 3 天: Evaluation (8h)
python scripts/run_config.py configs/ncfm/kvasir/ipc10_full.yaml \
  --stage evaluation \
  --load-path results/ncfm/kvasir/distilled_data

# 输出: results/ncfm/kvasir/distilled_data/data_20000.pt
```

---

## 四、快速验证 (Smoke Test)

### HoP-TM Smoke
```bash
# PathMNIST (IPC=1, Iteration=0) - 预计 5 分钟
python scripts/run_config.py configs/hop_tm/pathmnist/ipc1_smoke.yaml --stage buffer
python scripts/run_config.py configs/hop_tm/pathmnist/ipc1_smoke.yaml --stage distill
```

### NCFM Smoke
```bash
# PathMNIST (IPC=1, val_repeat=1) - 预计 10 分钟
python scripts/run_config.py configs/ncfm/pathmnist/ipc1_smoke.yaml --stage pretrain
python scripts/run_config.py configs/ncfm/pathmnist/ipc1_smoke.yaml --stage condense
python scripts/run_config.py configs/ncfm/pathmnist/ipc1_smoke.yaml \
  --stage evaluation \
  --load-path results/ncfm/pathmnist/distilled_data
```

---

## 五、常见问题

### Q1: Buffer 不存在怎么办？
```
FileNotFoundError: MTT/HoP-TM distill 需要当前配置对应的 expert buffer
```
**解决**: 先运行 `--stage buffer`

### Q2: NCFM pretrain 模型不存在？
```
FileNotFoundError: NCFM condense 需要预训练模型
```
**解决**: 先运行 `--stage pretrain`

### Q3: NCFM evaluation 找不到合成数据？
```
FileNotFoundError: NCFM evaluation 目录中没有 data_*.pt
```
**解决**: 先运行 `--stage condense`，或检查 `--load-path` 路径

### Q4: 如何查看中间结果？
- HoP-TM: `results/hop_tm/<dataset>/ipc10/checkpoints/`
- NCFM: `results/ncfm/<dataset>/distilled_data/data_*.pt`

---

**最后更新**: 2026-08-12  
**状态**: 流程文档完成，等待 GPU 资源执行实验
