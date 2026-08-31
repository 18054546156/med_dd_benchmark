# 距离实验配置

这些配置只覆盖距离实验的参数。数据路径、预训练教师模型和完整 NCFM 入口仍由具体运行工程提供。四种实验的核心字段如下：

```yaml
geometric_method: energy       # energy / sinkhorn / sliced_wasserstein
geometric_mode: replacement    # replacement 或 auxiliary
geometric_target_gradient_dose: 1.0
geometric_sinkhorn_epsilon: 0.1
geometric_sinkhorn_iterations: 200
geometric_sw_projection_count: 4096
geometric_sw_projection_seed: 1701
```

`replacement` 表示完全替换原始 CF；`auxiliary` 表示保留 CF，并按图像梯度剂量添加距离项。`0.0025` 表示距离项加权梯度目标约为 CF 梯度的 0.25%，不是两个 loss 数值的比例。

参数扫描建议：

- Energy auxiliary：`0.001, 0.0025, 0.005, 0.01, 0.02, 0.05`
- Sinkhorn：`epsilon=0.01, 0.03, 0.05, 0.10, 0.20, 0.50`，迭代 `100, 300, 1000`
- Sliced Wasserstein：投影数 `128, 512, 2048, 8192`，辅助剂量 `0.001, 0.005, 0.01, 0.05`

每次实验应保存完整 resolved config、代码版本、seed、训练日志和评估指标。
