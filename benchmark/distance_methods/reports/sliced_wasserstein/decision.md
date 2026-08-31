# SW 预实验审计结论

日期：2026-07-23  
对象：PathMNIST seed0、固定 `fixed_iid_U4096_unique` 的 `data_2000.pt`。本轮没有重新 condense，也没有使用测试集选方法。

## 核心结果

1. 历史 `K=64` Sliced-Wasserstein 的标量值看似稳定，但 synthetic-image 梯度高度依赖随机投影 bank；不能据此解释旧 CF+SW mixed 结果。
2. 梯度一致性随投影数增长：`K=2048` 类均值约 0.74，`K=4096` 约 0.86，`K=8192` 约 0.92。预注册门槛在 `K=4096/8192` 才通过。
3. matched-size real-real null 在所有审计 teacher/class 上通过，说明 SW 确实看到了 synthetic-specific 几何差异。
4. 稳定 `K=8192` 下，历史 `lambda_SW=0.05` 的加权 SW 梯度仅约为 CF 梯度的 0.2%，所以旧结果不能解释成“SW 权重太强”。
5. 等像素剂量的一步更新中，SW 在 frozen-teacher ridge/centroid 代理上明显正向；但 fresh ConvNet/ResNet18 训练代理没有稳定提升。
6. `SW residual` 在 ConvNet 为正、ResNet18 为负，不能证明 SW 提供了跨架构独立互补信息。

## 决策

当前不启动新的 `CF + SW` 2k/20k condensation，也不继续扫静态 `lambda_SW`。

现阶段被证实的不是“SW 已能提升 DD”，而是：

- 低投影数 SW 存在梯度估计问题；
- 几何 discrepancy 与 fresh-learner utility 之间仍有明显错位。

只有在高稳定投影或方差降低估计器下，SW 的更新能够在 fresh learner、双架构和错误方向 control 上稳定优于 base，才值得进入 condensation。

## 路径

- 远端：`/data/zengqiang/experiments/NCFM_SW_PREEXPERIMENT_AUDIT_20260723`
- 本地：`D:\Project\NCFM_Medmnist\NCFM_SW_PREEXPERIMENT_AUDIT_20260723`
- 分析 notebook：`SW_PREEXPERIMENT_ANALYSIS.executed.ipynb`
