# PathMNIST IPC10 学生归一化复评

## 评估修复

原 NCFM 的 CIFAR 分支在 `from_tensor=True` 时仍对学生输入执行 Normalize。
医疗适配分支过去跳过这一步，尽管保存的合成画布处于像素空间。
真实验证集和测试集都由 `utils/medical_dataset_utils.py` 执行 Normalize。
本次评估增加 `--synthetic_normalization once|raw`。默认 `once` 在解码后
应用与真实 val/test 相同的统计量，`raw` 复现旧学生输入。DSA 和 CutMix
在学生训练中照常执行。执行代码版本为 `73590b83f4df36abb3fb3a61f8c77bedfed50489`。

历史 ACC 来自两份不同的画布，且使用 HoP 的评估入口。

| HPC 作业 | 历史测试 ACC | `data_20000.pt` SHA256 |
|---|---:|---|
| `26128_0` | 78.0891% | `535d651f2204e6c70316f212b57b9961dac70cb2e45135032cf27c029e0be25f` |
| `26479_0` | 78.7967% | `9c885c049c1b56d6486abc6ad9ca6fa90e6a3875bd48bfab9f698399c7741c9f` |

`26128_0` 的数字来自本地 NCFM 仓库
`benchmark/audit_artifacts/table1_ipc10_snapshot/NCFM_evaluation_metrics.json`。
两份历史评估都记录了 `native_test_best`。HPC 的 `26479_0` 见
`results/variants/ncfm/PathMNIST/ConvNetD3_IN_W128/ipc10/seed0/slurm-26479_0/metrics.json`，
相对于 `/project/prj-sis01/xuxiaoyu/NCFM_Research`。其蒸馏配置
`batch_real=1024`、`load_memory=true`，正式间隔实验使用
`batch_real=128`、`load_memory=false`。A/B 保持画布固定，评估过程统一采用验证选模。

## 作业和产物

HPC 路径为 `/project/prj-sis01/xuxiaoyu/NCFM_Research/experiments/margin_study_20261126/code/outputs/normalization_reeval_20261127`。
账号为 `xiaoyuxu2`，`gpu-h100` 分区，`qos-high-gpu`，每任务一张 H100，
8 CPU、32G 内存，每个数组最多两个任务并行。

| 作业 | 内容 |
|---|---|
| `206297_0`、`206297_1` | 历史 `26479_0`，依次为 raw、once |
| `206297_2`、`206297_3` | 当前 `off/seed0`，依次为 raw、once |
| `206298_4` 至 `206298_11` | 其余八份画布按 once 评估，依赖 A/B 数组成功完成 |

日志为 `logs/eval-<job>_<task>.{out,err}`。每项在
`results/<artifact>_<normalization>/` 写出实际配置、输入哈希和五个学生的
验证选模 epoch、测试 ACC/BACC、逐类召回和混淆矩阵。
评估随机种子固定为 7341，蒸馏随机种子取已保存产物的 0、1、2。见 [运行前计划](PLAN_CN.md)。

## 间隔是否移动

取 2210 教师库的三个 checkpoint，与正式蒸馏教师文件的 SHA256 无交集。
从 PathMNIST 训练集每类独立抽取 512 张确定竞争类别，另取 512 张估计目标，
排除正式蒸馏预计算所用索引。固定教师测九份既有合成集，解码后每类 40 个视图。
表为跨种子、教师、类别平均绝对误差。

| 方法 | 竞争类 q10 | 竞争类 q25 | 竞争类 q50 | 最强竞争者 q10 |
|---|---:|---:|---:|---:|
| off | 0.02092 | 0.01493 | 0.01034 | 0.02540 |
| overall | 0.01678 | 0.01226 | 0.00862 | 0.01958 |
| rival | 0.01587 | 0.01121 | 0.00757 | 0.02020 |

rival 的三个配对种子在三个竞争类分位数的误差都下降。
竞争类 q10 的有符号差 `DD−real` 从 +0.01776 降到 +0.01162。
留出教师测得的真实最低 q10 类别依次为平滑肌（5）、结直肠腺癌上皮（8）、黏液（4）。
完整分位数、采样索引和教师哈希见 [诊断数据](margin_probe.json)。

## 代码与训练流程

基线使用 [NCFM 上游固定版本](https://github.com/gszfwsb/NCFM/tree/9e3e60b855fa918337013c8c3d460601690eb58e)
的 CF 损失和解码器。三个分支都关闭 SamplingNet，由 CF 损失采样 4096 个高斯频率。
`off` 在同一 PathMNIST 适配和配置下关闭辅助损失。两种间隔约束加入匹配更新，
原校准更新单独执行，见 [损失接入](../../../adapted/ncfm/condenser/compute_loss.py)
和 [角度分位数](../../../adapted/ncfm/NCFM/margin_matching.py)。

```text
每个冻结教师、每个真实类别
    用真实训练图统计竞争类别，计算目标 q10、q25、q50
每个蒸馏步、每个类别
    原解码器展开画布，用中间教师特征计算真实图与合成视图的 CF 损失
    overall 或 rival 用最终教师计算合成间隔，将 100 × 分位数 MSE 加入匹配损失
    更新合成画布，随后执行原校准更新
每份保存的画布、每个学生初始化
    解码并归一化一次，训练学生，用真实 val 选 epoch
    用所选学生在真实 test 记录 ACC、BACC 和逐类召回
```

## 优化日志

原作业 `203793` 中，off / overall / rival 各三项首次将 LR 显示为
`0.000000` 的步数分别为 `7560,8400,8160`、`7320,9080,8580`、
`6940,7700,7660`。日志精度为六位小数。调度器监控匹配与校准的总损失，
开启辅助项时匹配项含加权间隔损失。三个模式均出现低学习率，
这份日志无法识别间隔与校准的梯度关系。
