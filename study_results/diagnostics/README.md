# 诊断结果索引

这里保存的是 PathMNIST 训练集上的探索性统计，不是合成数据训练学生后的测试 ACC。

- `continuity_26479.csv`、`continuity_2210.csv`：每类、纹理原型、连续长度的计数与概率。
- `continuity_26479_summary.json`、`continuity_2210_summary.json`：数据、教师、合成张量哈希与采样索引。
- `margin_targets_metadata.json`：20 个教师、训练数据和固定目标文件的哈希及统计配置。目标张量留在 HPC。
- `smoke_*.json`：仅用于核对两步蒸馏、两轮学生训练的流程。数值不能用于比较算法。

角度间隔诊断数据位于 `../../scripts/coverage_results.json` 和
`../../scripts/coverage_teacher_fixedstats.json`。方法、解读和 HPC 路径见
`../../HANDOFF_NCFM_MARGIN_CN.md`。
