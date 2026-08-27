# 已核实结果

相对 fixed-U4096 NCFM，指标为 BACC 差值；Energy 的早期表同时记录 Macro-F1。

| 方法 | 结果 / 差值 |
|---|---|
| 原始 NCFM | BACC：81.054% / 86.756% |
| Energy Distance 替换 CF | ConvNet：+0.06pp；ResNet18：+0.34pp |
| CF + Energy | ConvNet：-0.133pp；ResNet18：-0.128pp |
| Sinkhorn Divergence 替换 CF | ConvNet：最高 +0.02pp；ResNet18：-2.82pp |
| Sliced Wasserstein 替换/辅助 | ConvNet：+0.145pp；ResNet18：+0.072pp |

这些是既有 PathMNIST 预实验结果，不是三个数据集、多随机种子的最终结论。当前没有证据表明这些方案稳定超过 fixed-U4096。
