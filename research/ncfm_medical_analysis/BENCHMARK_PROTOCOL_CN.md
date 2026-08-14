# NCFM / HoP 医学数据 Benchmark 协议

## 结论先行

正式 benchmark 必须同时保留两条结果轨：

1. **原始方法复现轨**：按各论文/官方实现的算法配置运行，回答“该方法按原设计能达到什么结果”。
2. **统一下游评估轨**：对已经生成的 synthetic data 使用同一个 evaluator，回答“在相同训练预算和测试协议下，哪个 synthetic data 更好”。

这两条轨不能互相替代，也不能把统一 evaluator 的参数反写进 NCFM 的 pretrain/condense 配置。

## NCFM 原始复现轨

以下参数属于 NCFM 方法本身，正式复现时保留并写入 manifest：

| 阶段 | 参数 | 正式设置 |
|---|---|---|
| Pretrain | teacher 数量 | 20 |
| Pretrain | 优化器 | teacher 训练使用官方实现的 SGD |
| Pretrain | 增强 | DSA + CutMix (`mixup: cut`) |
| Pretrain | medical 训练轮数 | 当前 adapted config 的显式值；不能靠日志猜测 |
| Condense | 目标 | NCFM CF loss |
| Condense | 频率数 T | 4096 (`num_freqs: 4096`) |
| Condense | sampling net | `false`，这是 baseline，不应偷偷改成 learned-frequency ablation |
| Condense | 迭代数 | 20000 |
| Condense | IPC | 10 |

`CutMix` 会改变 teacher 的训练轨迹和 teacher feature，因而属于完整方法的一部分。删除它会产生一个 `NCFM-DSA-only-pretrain` 消融实验，不能再把结果标成官方 NCFM baseline。

## HoP 原始复现轨

HoP 的 buffer、trajectory matching/distill、teacher/expert 数量和内部优化器按 HoP 代码及配置保留。HoP 不需要为了和 NCFM “表面一样”加入 CutMix；这会改变 HoP 的方法定义。

## 统一下游评估轨

NCFM 和 HoP 生成 synthetic data 后，必须交给同一个 evaluator。以下项目必须完全一致：

- 相同的 train/val/test split；
- 相同的 train-only mean/std 归一化；
- 相同的 synthetic tensor contract：`NCHW`、RGB、`[0,1]`、整数 label；
- 相同的 downstream optimizer、学习率、weight decay、scheduler、epoch、batch size；
- 相同的 augmentation policy；主表使用 `none`，DSA 作为单独敏感性分析；
- 相同的随机 seed 集合，至少 5 次；
- 相同的评估架构：至少 `ConvNet` 和 `ResNet18`；
- 相同的 test split 和指标计算方式。

当前控制协议为：

```yaml
optimizer: SGD
lr: 0.01
momentum: 0.9
weight_decay: 0.0005
epochs: 1000
batch_size: 256
augmentation: none
repeats: 5
architectures: [ConvNet, ResNet18]
```

这里的 SGD/1000 epoch 是**评估器训练协议**，不是 NCFM pretrain 或 condense 的官方配置。它也不是凭空宣称的“唯一标准”；如果最终论文选择 AdamW/2000 epoch，也必须让两个方法都重新跑同一协议，并在 manifest 中固定版本。最重要的是不要出现 NCFM 用一套 evaluator、HoP 用另一套 evaluator 的情况。

## 结果如何报告

主表至少包含：

| 方法 | 原始内部配置 | Eval 架构 | mean test acc | std | seed 列表 |
|---|---|---|---:|---:|---|
| NCFM | 原始复现轨 | ConvNet | 待实测 | 待实测 | 待实测 |
| NCFM | 原始复现轨 | ResNet18 | 待实测 | 待实测 | 待实测 |
| HoP | 原始复现轨 | ConvNet | 待实测 | 待实测 | 待实测 |
| HoP | 原始复现轨 | ResNet18 | 待实测 | 待实测 | 待实测 |

不得把旧日志中的 accuracy、示例数字或 toy 实验填入正式表。每个结果必须关联：config、command、代码 commit/SHA、数据统计 hash、synthetic hash、checkpoint、Slurm stdout/stderr。

## 脚本职责

- NCFM pretrain：只负责生成 20 个 teacher。
- NCFM condense：只负责生成显式指定的 `data_20000.pt`；禁止按 mtime 自动选择文件。
- HoP pipeline：只负责 HoP 自己的 buffer/distill 流程。
- `unified_eval_real.py`：只负责 NCFM/HoP 的统一下游训练和测试，不能修改原方法 config。

旧的 toy/legacy 入口不属于正式证据，尤其不能提交 `benchmark/run_all_phase1.sbatch` 和 `benchmark/run_phase1_experiments.sbatch` 来替代真实数据实验。
