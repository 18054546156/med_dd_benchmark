# NCFM 距离对照实验

本目录只管理四种距离实验：Energy Distance 替换 CF、CF 加 Energy、Sinkhorn Divergence 替换 CF，以及 Sliced Wasserstein 替换或辅助。原始 NCFM 代码仍位于 `raw/`，本目录不修改它。

## 代码

- `energy/loss.py`: `energy_distance(real, synthetic)`
- `sinkhorn/loss.py`: `sinkhorn_plan`、`entropic_ot`、`debiased_sinkhorn_divergence`
- `sliced_wasserstein/loss.py`: `projection_bank`、`exact_sliced_wasserstein`
- `cf_plus_energy.py`: 保留 CF 标量损失并加上 `energy_weight * energy_distance`

输入均为二维特征张量 `[样本数, 特征维数]`，输出为可反向传播的标量距离。它们是可嵌入 NCFM 的距离模块，不包含数据下载、预训练或完整蒸馏入口。

## 验证

```powershell
python -m pytest benchmark/distance_methods -q
python benchmark/distance_methods/run_distance_demo.py
```

## 结果记录

历史结果和判定见 `RESULTS.md`。新的服务器实验应将日志和合成数据写入仓库外的 `results/<method>/<dataset>/<seed>/`，不要提交数据文件。
