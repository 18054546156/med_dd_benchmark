# 医疗数据集蒸馏 Benchmark 当前状态

更新时间：2026-08-11
项目：`med_dd_benchmark`

## 1. 项目定位

本项目是一个可审计的医疗数据集蒸馏 Benchmark：

```text
raw/       原始算法仓库，只读，用于追溯作者代码和默认参数
adapted/  医疗数据适配代码，保留各算法自己的损失和训练流程
configs/  每个算法、每个数据集的迁移配置
scripts/  数据准备、配置分派和验证入口
data/     统一准备后的数据集和 manifest
results/  蒸馏结果、评估结果和运行日志
buffers/  MTT/HoP-TM 的专家轨迹 buffer
```

当前目标是比较 8 个方法在 3 个医疗数据集上的表现，不声称复现原论文在
PathMNIST、COVID 或 Kvasir 上的原始结果。

8 个方法为：`DC`、`DSA`、`DM`、`MTT`、`HoP-TM`、`NCFM`、`DataDAM`、`CAFE`。

## 2. 当前真实状态

| 层级 | 当前状态 | 可以得出的结论 |
|---|---|---|
| 统一数据合同 | 已完成 | 数据集名称、类别、尺寸、路径和标签格式已固定 |
| 三套 prepared 数据 | 已完成 | 有固定划分和 `manifest.json` |
| Loader 合同 | `24/24` 通过 | 8 个方法均能读取 3 个数据集 |
| One-step 核心计算 | `24/24` 通过 | 各方法的最小前向/反向或轨迹探针可执行 |
| 官方入口 smoke | `24/24` 通过 | 最小入口能够产生对应产物 |
| 完整 IPC=1/10/50 Benchmark | 未完成 | 不能使用 smoke 数字作为正式结果 |
| 原论文结果复现 | 不适用 | 这些医疗数据集不是多数原论文的原始实验设置 |

`smoke` 使用的是 `ipc=1`、少量迭代、少量 expert 或少量评估轮数，只证明
代码入口可以运行。它不证明收敛、最终准确率或方法优劣。

## 3. 数据合同

详细合同见 [`DATASET_CONTRACT.md`](DATASET_CONTRACT.md)。当前固定规格为：

| 数据集 | 输入 | 类别数 | 数据格式 | 归一化来源 |
|---|---:|---:|---|---|
| PathMNIST | `3 x 32 x 32` | 9 | MedMNIST NPZ | Benchmark 合同约定 |
| COVID | `3 x 112 x 112` | 4 | ImageFolder | ImageNet mean/std |
| Kvasir | `3 x 128 x 128` | 8 | ImageFolder | ImageNet mean/std |

COVID 和 Kvasir 的 ImageNet mean/std 是工程约定，不是这两个数据集官方发布的
统计量。三个数据集的类别、尺寸和标签合同由
`utils/medical_dataset_utils.py` 统一维护。

## 4. 每个方法的真实流水线

| 方法 | 正式流程 | 必须单独生成的前置产物 |
|---|---|---|
| DC | 真实图像 -> 梯度匹配 -> 合成图像 -> 独立评估 | 无 |
| DSA | 真实/合成图像 -> 可微 Siamese augmentation + 梯度匹配 -> 评估 | 无 |
| DM | 真实/合成图像 -> 类条件分布匹配 -> 合成图像 -> 评估 | 无 |
| MTT | 真实数据训练 expert -> 保存轨迹 -> 轨迹匹配蒸馏 -> 评估 | 每个数据集/模型一套 buffer |
| HoP-TM | FTD/GSAM expert -> 保存轨迹 -> 高阶轨迹匹配 -> 评估 | 每个数据集/模型一套 buffer |
| NCFM | 真实数据预训练 -> characteristic matching condense -> evaluation | 每个数据集的预训练模型 |
| DataDAM | attention/output matching -> 合成图像 -> 独立评估 | 无 |
| CAFE | 多层 feature alignment + inner loop -> 合成图像 -> 独立评估 | 无 |

对应的原始代码入口位于：

- DC/DSA：`raw/DatasetCondensation/main.py`
- DM：`raw/DatasetCondensation/main_DM.py`
- MTT：`raw/mtt-distillation/buffer.py`、`raw/mtt-distillation/distill.py`
- HoP-TM：`raw/HoP-TM/buffer/`、`raw/HoP-TM/distill/`
- NCFM：`raw/NCFM/pretrain/`、`raw/NCFM/condense/`、`raw/NCFM/evaluation/`
- DataDAM：`raw/DataDAM/main_DataDAM.py`
- CAFE：`raw/CAFE/distill.py`

## 5. 配置来源规则

当前配置参数按 `configs/CONFIG_PARAMETER_POLICY.md` 分为四类：

| 标签 | 含义 |
|---|---|
| `RAW` | 原始作者代码或原始 YAML 的默认值 |
| `CONTRACT` | 三个医疗数据集固有的类别、通道和输入尺寸 |
| `RESOURCE` | 为显存、CPU 或运行时间做的资源调整 |
| `MIGRATION` | 作者没有提供该医疗数据集配置时的迁移选择 |

`MIGRATION` 不能写成作者官方配置。`RESOURCE` 不能写成算法性能结论。
每个算法继续使用自己的参数体系，不能把 8 个方法强行改成同一套学习率、
迭代次数或 buffer 流程。

## 6. 正式 Benchmark 的完成标准

正式结果至少应覆盖：

```text
8 methods x 3 datasets x IPC={1,10,50} x fixed random seeds
```

每个实验单元必须保存：

- 实际执行的完整命令；
- 配置文件和解析后的运行参数；
- 当前 Git commit；
- 随机种子；
- 数据 `manifest.json` 或其 hash；
- 模型结构、输入尺寸、归一化和 augmentation；
- MTT/HoP-TM buffer 或 NCFM 预训练模型的来源；
- 合成数据和独立 test 集评估结果；
- 多次运行的 mean/std、运行时间和显存。

建议先完成 IPC=10，再扩展到 IPC=1 和 IPC=50。最终报告应使用独立 test
集，不使用合成数据自身的训练准确率作为 Benchmark 指标。

## 7. 运行入口

统一入口默认只打印命令，加 `--run` 才真正执行：

```powershell
# DC / DSA / DM
python scripts/run_config.py --config configs/dc_dsa_dm/pathmnist/ipc10_dc_full.yaml --run
python scripts/run_config.py --config configs/dc_dsa_dm/pathmnist/ipc10_dsa_full.yaml --run
python scripts/run_config.py --config configs/dc_dsa_dm/pathmnist/ipc10_dm_full.yaml --run

# MTT：先 buffer，再 distill
python scripts/run_config.py --config configs/mtt/covid/ipc10_full.yaml --algorithm mtt --stage buffer --run
python scripts/run_config.py --config configs/mtt/covid/ipc10_full.yaml --algorithm mtt --stage distill --run

# HoP-TM：先 buffer，再高阶轨迹蒸馏
python scripts/run_config.py --config configs/hop_tm/kvasir/ipc10_full.yaml --algorithm hop_tm --stage buffer --run
python scripts/run_config.py --config configs/hop_tm/kvasir/ipc10_full.yaml --algorithm hop_tm --stage distill --run

# NCFM：pretrain -> condense -> evaluation
python scripts/run_config.py --config configs/ncfm/pathmnist/ipc10_full.yaml --algorithm ncfm --stage pretrain --run
python scripts/run_config.py --config configs/ncfm/pathmnist/ipc10_full.yaml --algorithm ncfm --stage condense --run
python scripts/run_config.py --config configs/ncfm/pathmnist/ipc10_full.yaml --algorithm ncfm --stage evaluation --load-path <合成数据目录> --run
```

## 8. 目前不能作出的结论

当前不能声称：

1. 8 个算法已经完成三个数据集上的完整 Benchmark；
2. smoke 的准确率就是论文结果；
3. 三个数据集的 mean/std 都是官方统计值；
4. 已经复现了原论文在这些医疗数据集上的结果；
5. HoP-TM、MTT 或 NCFM 的最小入口已经证明正式规模能够收敛。

当前最准确的项目状态是：

> 医疗数据合同、算法适配、loader、核心 one-step 和最小入口已经验证；
> 正式 Benchmark 仍需按各算法的完整 pipeline 分别运行和评估。
