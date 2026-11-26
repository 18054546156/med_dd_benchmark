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
| [study_results/margin_20261126](study_results/margin_20261126/) | PathMNIST 正式评估的原始数据、配置和汇总 |
| [benchmark](benchmark/) | benchmark 工具和参考内容 |

正式比较已完成。竞争类别间隔匹配比整体间隔匹配高 1.15 pp ACC，
比 NCFM 对照低 1.39 pp。真实数据、checkpoint 和合成数据保留在 HPC。
