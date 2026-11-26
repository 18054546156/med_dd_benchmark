# PathMNIST 类间间隔正式实验归档

Slurm 数组 `203793` 的九项任务均以 `COMPLETED|0:0` 结束。每项使用一张 H100 NVL，
在 `hpcgpu111` 或 `hpcgpu112` 上运行。北京时间 2026-11-27 00:47 至 03:58。

| 文件 | 内容 |
|---|---|
| [summary.json](summary.json) | 按蒸馏种子聚合的 ACC、BACC 和配对差值 |
| `{off,overall,rival}_seed{0,1,2}.json` | 45 次学生评估，含选中 epoch、逐类召回及混淆矩阵 |
| `*_config.yaml` | 实际运行时配置，模式由数组任务号和批处理脚本决定 |
| [slurm_accounting.txt](slurm_accounting.txt) | 登录节点查询的 Slurm 完成状态与耗时 |
| [synthetic_sha256.txt](synthetic_sha256.txt) | HPC 上九个合成数据文件的相对路径和 SHA256 |
| [training_code_sha256.txt](training_code_sha256.txt) | HPC 训练入口、损失、配置和评估代码的 SHA256 |
| [targets_metadata.json](targets_metadata.json) | 教师、训练数据和预计算目标的哈希 |

批处理脚本中 `task_id // 3` 分别选择 off、overall、rival，
`task_id % 3` 对应蒸馏种子。辅助损失权重为 100。正式运行代码与 GitHub
`ca9834b` 的训练和评估文件核对一致。运行日志和合成张量路径见
[Handoff](../../HANDOFF_NCFM_MARGIN_CN.md)。

结果来自 HPC 登录节点重跑汇总，九份评估均由 7,180 张测试图的
9 类混淆矩阵复算验证。结果解读见 [实验结果](../RESULTS_20261126_CN.md)。
