# Phase 2：NCFM 新方法验证协议

Phase 2 只在 Phase 1 的真实 artifact gate 通过后执行。每个方法都必须使用
明确的 train-only statistics、固定的 NCFM teacher checkpoint、明确的 synthetic
artifact、独立 seed 和独立日志。机制实验结果不能直接写成 NCFM 已确认缺陷。

## 三个真实方法变体

### P2.1 Scrambled-QMC frequency estimator

使用 scrambled Sobol 点经过正态逆 CDF 得到频率，和同一批真实/synthetic 特征
上的 Gaussian Monte Carlo 估计配对比较。每个 `T` 至少 20 个独立 scramble，报告
均值、标准差、相对方差下降和与高预算 reference 的误差。

只有在把该 estimator 真正接入 NCFM condenser、重新生成 synthetic data 并完成
统一下游评估后，才能讨论它是否改善最终 test accuracy。当前 Phase 2 分析入口
先验证 estimator 本身，不修改 baseline `num_freqs=4096`。

### P2.2 Exact importance sampling

令目标频率分布为 `p=N(0,I)`，提议分布为 `q=N(mu,I)`，使用未截断的 `p/q` 权重。
必须报告 ESS、权重最大值和估计方差。禁止把 clipped weights 称为无偏；若以后
需要 clipping，必须作为独立的 biased-stabilized ablation。

### P2.3 Learned-frequency sampler

启用 NCFM 的 `sampling_net=true`，其余 teacher、IPC、`niter=20000`、评估协议
保持不变。该变体必须使用新的 `RUN_ID`，不能写入 baseline 目录。它是实际的
condenser 方法变体，不能只用 estimator 数值模拟代替。

## 共同的 Holdout 证书分析

对 baseline 和上述三个变体，在固定 synthetic artifact 上使用未参与任何优化的独立频率 bank，按每频率误差
`|phi_real(w)-phi_syn(w)|^2 <= 4` 计算经验 Bernstein 上界。该证书只对“固定的
feature pair + 独立 frequency distribution”条件成立，不能自动覆盖优化过程、
换 backbone 或换数据集。

因此 Holdout 证书是三种方法的共同泛化诊断，不计作第四种方法，也不能替代
P2.3 的真实 condenser 重跑和统一 downstream evaluation。

## 结论等级

`complete` 只表示测量完成；`supported` 表示观察到预注册效果；`confirmed` 仍需
对应的真实 condenser 重跑、统一 Eval、足够 seeds 和 artifact SHA256。缺少输入
时必须输出 `insufficient_evidence`，不能回退到 toy 数据或旧日志。
