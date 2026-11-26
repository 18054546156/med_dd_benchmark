# Medical Dataset Distillation Benchmark

本分支在医疗图像数据蒸馏 benchmark 中研究 NCFM 的类间角度间隔匹配。
完整实验接手说明、实际诊断数据和 HPC 作业状态见
[中文 handoff](HANDOFF_NCFM_MARGIN_CN.md)。

本轮工作入口见 [24 小时计划](study_results/PLAN_20261126_CN.md)、
[结果记录](study_results/RESULTS_20261126_CN.md)和
[论文草稿](paper_materials/ncfm_rival_margin_draft.md)。

| 目录 | 内容 |
|---|---|
| [adapted/ncfm](adapted/ncfm/) | NCFM 数据适配与可选的间隔损失 |
| [configs](configs/) | 数据集、算法和实验配置 |
| [scripts](scripts/) | 目标预计算、训练、评估与结果汇总脚本 |
| [study_results/diagnostics](study_results/diagnostics/) | 已完成的冻结教师诊断和短测记录 |
| [benchmark](benchmark/) | benchmark 工具和参考内容 |

**当前没有新算法的正式 ACC/BACC。** 正式作业运行状态以 Slurm 为准；
短测用于验证流程，不能作为算法效果。真实数据、checkpoint 和合成数据
保留在 HPC，不存入本仓库。
