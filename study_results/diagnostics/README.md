# 诊断结果索引

这里保存的是 PathMNIST 训练集上的探索性统计，不是合成数据训练学生后的测试 ACC。

- `continuity_26479.csv`、`continuity_2210.csv`：每类、纹理原型、连续长度的计数与概率。
- `continuity_26479_summary.json`、`continuity_2210_summary.json`：数据、教师、合成张量哈希与采样索引。
- `margin_targets_metadata.json`：20 个教师、训练数据和固定目标文件的哈希及统计配置。目标张量留在 HPC。
- `smoke_*.json`：仅用于核对两步蒸馏、两轮学生训练的流程。数值不能用于比较算法。
- `smoke_h100_204035_{off,overall,rival}.json`：上述流程在同一张 H100 NVL 上的顺序短测产物。
- `rival_margin_crossbank.json`：两份合成集、每份三位留出教师的逐类角度间隔。
- `rival_margin_crossbank_decode.json`：另含同批真实图裁切放大前后的配对统计。

角度间隔诊断数据位于 `../../scripts/coverage_results.json` 和
`../../scripts/coverage_teacher_fixedstats.json`。方法、解读和 HPC 路径见
`../../HANDOFF_NCFM_MARGIN_CN.md`。
