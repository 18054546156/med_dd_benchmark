# NCFM 类间间隔实验交接

更新日期为 2026-11-27（Asia/Shanghai）。这里的任务是研究 PathMNIST IPC=10
的合成数据能否更好地覆盖易混类别，保留原始 NCFM 作为对照。

## 一眼看懂目前状态

**正式评估 9/9 已完成。** NCFM 对照、整体间隔匹配、竞争类别间隔匹配的
测试 ACC 分别为 49.97%、47.44%、48.59%。竞争类别匹配相对 NCFM 的
配对 ACC 差为 −1.39 pp。每组包含 3 个蒸馏种子和每种子 5 次学生评估。

本轮资料：[24 小时执行计划与实验判据](study_results/PLAN_20261126_CN.md)、
[诊断和正式结果状态](study_results/RESULTS_20261126_CN.md)、
[论文草稿](paper_materials/ncfm_rival_margin_draft.md)。

GitHub 仓库：<https://github.com/18054546156/med_dd_benchmark>

实验分支：`codex/ncfm-margin-study`，代码在该分支，没有合并到 main。
本地工作树：`C:/Users/Administrator/Documents/project/ncfm_margin_study`。
HPC 已部署代码和产物的目录：

```text
/project/prj-sis01/xuxiaoyu/NCFM_Research/experiments/margin_study_20261126/code
```

## 现有实验发现

两份已完成的 PathMNIST 合成集在未使用对应教师库的教师上评估。
两组 20 个教师 checkpoint 的文件 SHA256 无交集，但这一点不能证明它们的
训练随机种子或模型来源完全独立。

| 被评估 DD | 留出教师的真实样本角度间隔低分位 | 留出教师的 DD 角度间隔低分位 |
|---|---:|---:|
| 26479_0 | 0.0537 | 0.0731 |
| 2210 | 0.0437 | 0.0692 |

表中为每类第 10 百分位再对 9 类取平均，是真实训练图和 DD 的教师特征
诊断，**不是测试 ACC**。同类最近邻余弦距离总体相近。采样为每类 192 张
真实查询、8 组每类 40 张真实参照，并与每类 40 个 DD 解码视图比较。
原始逐类数据分别在 `scripts/coverage_teacher_fixedstats.json` 和
`scripts/coverage_results.json`；脚本是 `scripts/coverage_probe.py`。

之前的纹理连续性实验中，真实图经过 factor=2 裁切放大后，平均连续概率
变化 5.21 个百分点；两个 DD 解码视图与真实原图的差距约 1.38、1.37
个百分点。它说明解码是重要混杂因素，不能据此证明连续性损失能提升 ACC。
逐类结果和索引在 `study_results/diagnostics/`。

## 方法和公平对照

三组都使用 NCFM CF loss，保存 9 类×10 张 `32×32` 画布。原版
`factor=2` 将其解码成每类 40 个视图，共 360 个。这是存储 IPC=10，
学生实际视图 IPC=40；比较时不要混同。

| 模式 | 配置值 | 改动 |
|---|---|---|
| 原版对照 | `NCFM_MARGIN_MODE=off` | 无额外间隔损失 |
| 整体间隔 | `overall` | 匹配正确类和最强竞争类的角度间隔分位数 |
| 竞争类别 | `rival` | 每类按真实训练图的 logits 找两个竞争类，分别匹配角度间隔分位数 |

预计算目标在每位冻结教师下使用每类 512 张真实训练图，随机种子 7341，
分位数为 10%、25%、50%。优化时额外损失权重为 100，是**尚未调优的探索值**。
目标张量经解码器对合成画布反向传播，教师参数冻结。

正式配置为每组 3 个蒸馏种子、20,000 次蒸馏，每份合成集进行 5 次
学生训练，每次 2,000 epoch。使用验证集选 epoch，随后测试集计算 ACC、
BACC、逐类召回和混淆矩阵；超参数不能根据测试结果挑选。
实现不改写旧实验目录，`off` 分支可直接运行。代码入口：

- `adapted/ncfm/NCFM/margin_matching.py`：辅助损失。
- `adapted/ncfm/condenser/compute_loss.py`：结合原 CF loss。
- `scripts/precompute_margin_targets.py`：从训练集预计算目标。
- `scripts/margin_run.sbatch`、`scripts/write_margin_config.py`：正式任务和配置。
- `scripts/collect_margin_results.py`：按蒸馏种子汇总，不把 5 次学生初始化当成 5 个蒸馏种子。

目标采用旧 26479 教师库当时的 mean `[0.741, 0.533, 0.706]`、std
`[0.402, 0.821, 0.407]`，用于保持教师输入一致；**这不应称为
PathMNIST 的实际逐通道标准差**。2210 教师采用另一套已计算的训练集统计。
跨教师诊断分别使用对应标准化。正式方法只用 26479 教师库。

## 任务状态和接手命令

下表是上述更新时间的快照，不代表实时状态。

| 作业 | 作用 | 快照 |
|---|---|---|
| 203745 | 最终目标预计算 | COMPLETED，1 张 RTX 4090 |
| 203721 | 三组短测 | COMPLETED |
| 203747 | 最终竞争类选择短测 | COMPLETED |
| 203789 | H100 并发短测 | 3 项 FAILED，108 节点 CUDA 设备忙或不可用 |
| 204035 | H100 单卡顺序短测 | COMPLETED，0:0，112 节点 H100 NVL，3 分 13 秒 |
| 203793 | 9 项正式数组 | 全部 COMPLETED，0:0，11 月 27 日 00:47 至 03:58 |
| 204052 | 核验、汇总并推送 | FAILED，计算节点账务查询失败，推送返回 403 |

使用 Windows SSH 别名：

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 luhpc
sacct -j 204035,203793,204052 -n -P --format=JobID,State,Elapsed,NodeList
cd /project/prj-sis01/xuxiaoyu/NCFM_Research/experiments/margin_study_20261126/code
ls -lh logs/margin-smoke-204035* logs/margin-203793* logs/margin-collect-204052* 2>/dev/null
less outputs/collected_203793_login_20261127/summary.json
```

正式结果在 `outputs/formal_h100/{off,overall,rival}/seed{0,1,2}/`，
每项包含运行配置、蒸馏产物及 `evaluation.json`。目标文件在
`outputs/margin_targets_logits_v2.npz`；对应元数据已纳入本仓库
`study_results/diagnostics/margin_targets_metadata.json`。数据集和模型权重
保留在 HPC，不提交到 GitHub。

收集任务先查 Slurm 退出状态，并根据 9 类测试混淆矩阵复算 ACC/BACC，
再按蒸馏种子生成两组配对差。204052 的故障日志仍在
`logs/margin-collect-204052.err`，成功重算的 HPC 汇总目录为
`outputs/collected_203793_login_20261127/`。登录节点能查询正常账务，
九份结果已归档至 [GitHub 分支的正式实验目录](study_results/margin_20261126/)。
新版收集器可读取从登录节点导出的 `sacct` 快照，并在账务不可用、
结果未全部验证或 GitHub 拒绝权限时以失败状态退出。

正式训练沿用 `98dc0cb` 的运行脚本，HPC 脚本 SHA256 为
`bed7a1a04e2e4f5aa7aedfb2174e7d769b7f55e355984efc60d17f753d59575c`。
旧收集作业 204052 的 Slurm 脚本 SHA256 为
`60e0466ed82422931cbd12e1572bc8bc15022c5e126a2d2a99d7f5eb88595017`。
204035 的三组短测产物位于 `outputs/smoke_h100_204035/`，每组均有
`data_2.pt` 和 `evaluation.json`。203793 在 H100 NVL 节点
`hpcgpu111` 和 `hpcgpu112` 运行，单项用时 53 至 58 分钟。

## 已知故障与保留的现场

- `meddd` 当前 PyTorch 为 CUDA 13 构建，4090 节点驱动不兼容。当前脚本
  改用已有的 `/home/xiaoyuxu2/.conda/envs/hopbench-cu128/bin/python`。
- 原入口曾覆盖 Slurm 的 `CUDA_VISIBLE_DEVICES`，多任务挤到同一 GPU。
  当前代码只在环境未设置时使用命令行 GPU 参数。
- 作业 203749 的 9 项因此失败，结果目录 `outputs/formal/` 只含失败运行
  的配置，不是正式结果；新数组使用 `outputs/formal_h100/`。
- 4090 节点的 Slurm 空闲卡存在其他 GPU 进程占用。脚本检查实测空闲显存
  至少 18 GiB，未通过即退出。203762 的一项卡住后已停止。不要解除保护
  或修改其他人的任务。
- 原始 `26128_0` 合成文件暂未在已检查的 HPC 归档中找到；本研究使用
  26479_0 和 2210 两份已核验哈希的产物，不能将其混称为历史 78.09% 的
  同一份合成集。

## 下一步

1. 确认本次 NCFM 对照与历史 `26479_0` 的预处理、学生评估配置及选模差异。
2. 当前权重 100 的两组辅助目标均低于 NCFM 对照。按既定判据，
   权重 100 的方案停止进入独立确认实验。
