# 配置参数政策与来源审计

本文档是 `configs/` 下所有 YAML 的共同依据。它把参数分成四类，避免把
作者默认值、医疗数据迁移值和显存妥协值混写成一个“官方配置”。

## 1. 参数来源分层

| 标签 | 含义 | 允许如何修改 |
| --- | --- | --- |
| `RAW` | 原作者仓库的命令行默认值或原始 YAML 值 | 算法复现实验优先保持不变 |
| `CONTRACT` | 本 benchmark 的统一数据合同 | 只描述数据集名称、类别数、通道和输入尺寸 |
| `RESOURCE` | 为当前显存、CPU 或评估时间做的资源适配 | 可以改 batch、worker、评估次数，但必须在 YAML 中说明 |
| `MIGRATION` | 作者没有提供这三个医疗数据集时的迁移选择 | 必须写出参考来源，不能称为作者官方值 |

`ipc10_full.yaml` 是正式实验起点：核心损失、更新规则、学习率和迭代预算尽量
按 `RAW`。`ipc1_smoke.yaml` 只验证数据加载、模型前向、反向和文件输出，不产生
论文结果。所有 full 配置都不是作者在 PathMNIST、COVID 或 Kvasir 上发布的官方结果。

## 2. 医疗数据合同

| 数据集 | `channel` | `im_size` | `num_classes` | 尺寸来源 |
| --- | ---: | --- | ---: | --- |
| PathMNIST | 3 | `[32, 32]` | 9 | MedMNIST 原始图像为 28x28；HoP-TM 的 PathMNIST 配置和 ConvNet 兼容性使用 32x32 |
| COVID | 3 | `[112, 112]` | 4 | HoP-TM 作者的 COVID 配置使用 112x112；原始 X 光图像尺寸不固定 |
| Kvasir | 3 | `[128, 128]` | 8 | Kvasir 原始图像尺寸不固定；128 是参考 HoP-TM/ImageNet 子集实验的迁移选择，不是 Kvasir 官方尺寸 |

这三个字段由 `scripts/run_config.py` 校验。图像在共享数据层 resize，算法只接收
`[C, H, W]` 的 float tensor 和标量 LongTensor 标签。

## 3. 各算法的 raw 基准

### DC / DSA

原始来源：`raw/DatasetCondensation/main.py`。

```text
Iteration=1000, lr_img=0.1, lr_net=0.01
batch_real=256, batch_train=256, init=noise
DC:  dsa_strategy=None
DSA: dsa_strategy=color_crop_cutout_flip_scale_rotate
```

`main.py` 根据 `method` 设置 DSA 开关：`method=DC` 时不启用 DSA，
`method=DSA` 时启用。因此 DC YAML 不应再写 `dsa: true`。大图数据集只允许
降低 batch 作为 `RESOURCE`，不能因为像素数增加就把 `lr_img` 擅自改成 10 或 100。

### DM

原始来源：`raw/DatasetCondensation/main_DM.py`。

```text
Iteration=20000, lr_img=1.0, lr_net=0.01
batch_real=256, batch_train=256, init=real
dsa_strategy=color_crop_cutout_flip_scale_rotate
```

这里的增强策略是 raw 默认值。把它设成 `None` 会改变 DM 的输入处理流程，不能
仅为了和 DC 配置看齐而关闭。

### MTT

原始来源：`raw/mtt-distillation/distill.py` 和 `raw/mtt-distillation/buffer.py`。

```text
distill: Iteration=5000, lr_img=1000, lr_lr=1e-5
         lr_teacher=0.01, expert_epochs=3, syn_steps=20
         max_start_epoch=25, batch_real=256, batch_train=256
buffer:  num_experts=100, train_epochs=50
         batch_real=256, batch_train=256, lr_teacher=0.01
```

MTT 的 `lr_img=1000` 是作者代码的默认值，不能按 DC 的 0.1 或普通 SGD 直觉
替换成 1、10、100。COVID/Kvasir 只可把 batch 标为 `RESOURCE` 并降低；专家轨迹
仍必须针对每个数据集、模型、输入尺寸和归一化重新生成。

### HoP-TM

原始来源：`raw/HoP-TM/exp_configs/PathMnist/` 和
`raw/HoP-TM/exp_configs/COVID/`。

PathMNIST IPC10 的 raw 参考为：`ConvNet`、`lr_img=10`、`lr_teacher=0.01`、
`syn_steps=80`、`expert_epochs=2`、`Iteration=10000`、`base_threshold=5e-9`、
`growing_factor=1.5`、`lamb=0.5`、`max_start_epoch=4`。

COVID IPC10 的 raw 参考为：`ConvNetD5`、`lr_img=100`、`lr_teacher=0.001`、
`syn_steps=80`、`expert_epochs=2`、`Iteration=10000`、`base_threshold=1e-10`、
`growing_factor=1.3`、`lamb=1.0`、`max_start_epoch=20`。

Kvasir 没有作者 YAML，使用 `ConvNetD5` 和 COVID/128x128 ImageNet 子集经验作为
`MIGRATION`，不是官方配置。HoP-TM 也必须先生成与 distill 完全匹配的 buffer。

### DataDAM

原始来源：`raw/DataDAM/main_DataDAM.py`。

```text
Iteration=20000, lr_img=1, lr_net=0.01
batch_real=64, batch_train=64, init=real
dsa_strategy=color_crop_cutout_flip_scale_rotate, task_balance=0.01
```

DataDAM 的 raw 入口没有 `attention.attn_layers`、`extract_attention.method` 等
命令行参数。当前 YAML 中若保留这些字段，它们只能作为说明性记录，不能声称会
改变运行过程；真正生效的字段以 `scripts/run_config.py` 映射表为准。

### CAFE

原始来源：`raw/CAFE/distill.py`。

```text
Iteration=2000, lr_img=0.1, lr_net=0.01
batch_real=256, batch_train=256, init=noise, dsa_strategy=None
first_weight=1.0, second_weight=1.0
third_weight=0.1, fourth_weight=0.1, inner_weight=0.01
lambda_1=0.04, lambda_2=0.03
```

`cafe.alpha/beta/feature_layers` 不是当前 raw CLI 参数。保留时必须标注为说明性
字段；统一入口只传递 raw 入口能接收的 top-level 参数。

### NCFM

原始来源：`raw/NCFM/config/`。NCFM 的正式流程是：

```text
pretrain -> condense -> evaluation
```

PathMNIST 使用 CIFAR10 参考的 `size=32/depth=3`；COVID 使用合同的
`size=112`，网络深度 4 是从 TinyImageNet 64 分辨率配置迁移；Kvasir 使用合同的
`size=128/depth=5`，参考 ImageNette 128 分辨率配置。`batch_real`、`batch_size`、
`workers` 在本机是 `RESOURCE` 参数，不能被误写成作者在医疗数据上的最优值。

## 4. 哪些 YAML 字段真正生效

统一入口不是通用的“把所有 YAML 键自动转成 CLI”。
`scripts/run_config.py` 按算法显式映射参数，以保证不同算法的流程不被强行统一。

| 算法 | 有效入口 | 阶段 |
| --- | --- | --- |
| DC / DSA | `adapted/dc_dsa_dm/main.py` | distill |
| DM | `adapted/dc_dsa_dm/main_DM.py` | distill |
| MTT | `adapted/mtt/buffer.py`、`adapted/mtt/distill.py` | buffer -> distill |
| HoP-TM | `adapted/hop_tm/buffer/buffer_FTD.py`、`adapted/hop_tm/distill/distill_high_order_spl.py` | buffer -> distill |
| NCFM | `adapted/ncfm/pretrain/pretrain_script.py`、`condense/condense_script.py`、`evaluation/evaluation_script.py` | pretrain -> condense -> evaluation |
| DataDAM | `adapted/datadam/main_DataDAM.py` | distill |
| CAFE | `adapted/cafe/distill.py` | distill |

每个配置的 `network`、`optimizer`、`attention` 或 `cafe` 嵌套块如果没有出现在
该算法的映射表中，就只是实验记录，不会自动改变 raw 入口。正式报告应记录 raw
默认、迁移修改、资源修改和验证结果四列。
