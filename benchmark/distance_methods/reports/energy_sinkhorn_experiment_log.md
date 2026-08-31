# Energy 与 Sinkhorn：同一个例子的数学和代码说明

本文使用同一组二维特征，解释 Energy distance、Sinkhorn/OT、`epsilon` 和 `0.25%` 梯度剂量。

## 1. 统一例子

```python
real = torch.tensor([
    [0.0, 0.0],
    [2.0, 0.0],
])

synthetic = torch.tensor([
    [1.0, 0.0],
    [3.0, 0.0],
])
```

```text
real.shape      = [2, 2]
synthetic.shape = [2, 2]
```

这里第一个数字是样本数量，第二个数字是特征维度：

```text
X1 = (0, 0)       Y1 = (1, 0)
X2 = (2, 0)       Y2 = (3, 0)
```

示意图：

```text
真实：     X1 -------- X2
           0           2

合成：        Y1 -------- Y2
              1           3
```

实际 NCFM 中，真实和合成的样本数可以不同：

```text
real      = [N_real, D]
synthetic = [N_syn, D]
```

例如真实 batch 是 `[1024, 128]`，合成图像是 `[10, 128]`，仍然可以计算距离。两者的 `D` 必须相同。

## 2. `torch.cdist` 做什么？

`p=2` 表示欧氏距离：

$$
d(a,b)=\sqrt{\sum_k(a_k-b_k)^2}
$$

### 2.1 真实-合成距离

```python
cross = torch.cdist(real, synthetic, p=2)
```

它计算每个真实点到每个合成点的距离：

$$
cross=
\begin{bmatrix}
d(X_1,Y_1)&d(X_1,Y_2)\\
d(X_2,Y_1)&d(X_2,Y_2)
\end{bmatrix}
=
\begin{bmatrix}
1&3\\
1&1
\end{bmatrix}
$$

例如：

$$
d(X_1,Y_1)=\sqrt{(0-1)^2+(0-0)^2}=1
$$

### 2.2 真实内部距离

```python
rr = torch.cdist(real, real, p=2)
```

这里不是只计算一个距离，而是第一份 `real` 作为行、第二份 `real` 作为列：

$$
rr=
\begin{bmatrix}
d(X_1,X_1)&d(X_1,X_2)\\
d(X_2,X_1)&d(X_2,X_2)
\end{bmatrix}
=
\begin{bmatrix}
0&2\\
2&0
\end{bmatrix}
$$

### 2.3 合成内部距离

```python
ss = torch.cdist(synthetic, synthetic, p=2)
```

同理：

$$
ss=
\begin{bmatrix}
0&2\\
2&0
\end{bmatrix}
$$

## 3. Energy distance

### 3.1 数学思想

Energy 不建立点到点运输计划，而是比较三类距离：

```text
真实点到合成点
真实点到真实点
合成点到合成点
```

公式：

$$
D_E(P,Q)=2E\|X-Y\|-E\|X-X'\|-E\|Y-Y'\|
$$

### 3.2 代入例子

跨分布距离：

$$
E\|X-Y\|=(1+3+1+1)/4=1.5
$$

真实内部距离：

$$
E\|X-X'\|=(0+2+2+0)/4=1
$$

合成内部距离：

$$
E\|Y-Y'\|=(0+2+2+0)/4=1
$$

因此：

$$
D_E=2\times1.5-1-1=1
$$

### 3.3 对应代码

文件：

```text
distance_benchmark/energy/loss.py
```

```python
# 真实特征到合成特征的两两距离。
cross = torch.cdist(real, synthetic, p=2).mean()

# 真实特征内部的两两距离。
rr = torch.cdist(real, real, p=2)

# 合成特征内部的两两距离。
ss = torch.cdist(synthetic, synthetic, p=2)

# Energy distance。
loss = (
    2.0 * cross
    - rr.mean()
    - ss.mean()
)
```


### Energy 短 

| Energy 实验 | ConvNet | ResNet18 | 失败含义 |
|---|---:|---:|---|
| 2k-step，Energy 替换 CF | `+0.06 / -0.29` | `+0.34 / +0.28` | ConvNet F1 已负，不能替换 CF |
| 2k-step，CF + Energy 0.25% | `-1.05 / -1.80` | `+0.25 / +0.35` | ConvNet 明显受损，剂量不合适 |
| 2k-step，CF + Energy 0.5% | `-0.13 / -0.19` | `+0.80 / +0.67` | 架构分裂 |
| 2k-step，CF + Energy 1% | `+0.23 / +0.27` | `+0.41 / +0.52` | 仅短程候选信号 |

### Energy full20k 恢复状态

| PathMNIST full20k | ConvNet BACC | ResNet18 BACC |
|---|---:|---:|
| fixed-U4096 | 81.054% | 86.756% |
| NCFD + Energy 1% | 80.921% | 86.628% |
| 差值 | **-0.132pp** | **-0.128pp** |

这组 full20k 数值同样只作为旧 ledger 的失败解释，不作为正式结果引用。Energy 1% 没有超过 fixed-U4096，两个架构均略微下降。

## 4. Sinkhorn / Optimal Transport

### 4.1 数学思想

OT 要决定：真实点向哪些合成点运输多少质量。

运输计划记为：

$$
\pi_{ij}=
\text{从真实点 }X_i\text{ 向合成点 }Y_j\text{ 运输的质量}
$$

本例的计划矩阵是：

$$
\pi=
\begin{bmatrix}
\pi_{11}&\pi_{12}\\
\pi_{21}&\pi_{22}
\end{bmatrix}
$$

因为有两个真实点和两个合成点，每个点的质量是 `1/2`，所以：

$$
\sum_j\pi_{ij}=0.5,
\qquad
\sum_i\pi_{ij}=0.5
$$

前者是行边际，后者是列边际。






### 4.2 运输成本

代码：

```python
def _cost(left, right):
    return 0.5 * torch.cdist(
        left,
        right,
        p=2,
    ).square()
```

数学定义：

$$
C_{ij}=\frac12\|X_i-Y_j\|_2^2
$$

本例距离矩阵是：

$$
\begin{bmatrix}1&3\\1&1\end{bmatrix}
$$

所以成本矩阵是：

$$
C=\begin{bmatrix}0.5&4.5\\0.5&0.5\end{bmatrix}
$$

例如：

$$
C_{12}=\frac12\times3^2=4.5
$$

成本越高，运输计划越不愿意使用这条路线。

### 4.3 epsilon 是什么？

Sinkhorn 加入熵正则：

$$
\min_\pi
\langle\pi,C\rangle
+\varepsilon\sum_{i,j}\pi_{ij}(\log\pi_{ij}-1)
$$

`epsilon` 控制运输计划的平滑程度。

代码：

```python
log_kernel = -cost / float(epsilon)
```

对应：

$$
K_{ij}=\exp(-C_{ij}/\varepsilon)
$$

```text
epsilon 小，例如 0.05：
更偏向最低成本路线；计划更尖锐；更接近原始 OT；可能不稳定。

epsilon 大，例如 0.20：
允许更多路线分配质量；计划更平滑；通常更稳定；可能过度平均。
```

### 4.4 Sinkhorn 迭代

```python
for _ in range(iterations):
    # 调整行，使每个真实点运出的质量正确。
    log_u = log_a - torch.logsumexp(
        log_kernel + log_v.unsqueeze(0),
        dim=1,
    )

    # 调整列，使每个合成点收到的质量正确。
    log_v = log_b - torch.logsumexp(
        log_kernel + log_u.unsqueeze(1),
        dim=0,
    )
```

最终：

```python
plan = torch.exp(
    log_kernel
    + log_u.unsqueeze(1)
    + log_v.unsqueeze(0)
)
```

此时：

```text
plan.shape = [真实样本数, 合成样本数]
plan[i, j] = 真实点 i 向合成点 j 运输的质量
```

### 4.5 去偏 Sinkhorn

普通熵正则 OT 可能存在自偏差，因此使用：

$$
S_\varepsilon(P,Q)=
OT_\varepsilon(P,Q)
-\frac12OT_\varepsilon(P,P)
-\frac12OT_\varepsilon(Q,Q)
$$

代码：

```python
loss = (
    entropic_ot(real, synthetic)
    - 0.5 * entropic_ot(real, real)
    - 0.5 * entropic_ot(synthetic, synthetic)
)
```


### Sinkhorn 短

| 100-step Sinkhorn-only replacement | ConvNet BACC / F1 | ResNet18 BACC / F1 |
|---|---:|---:|
| epsilon=0.05 | `-3.43 / -5.39` | `-10.28 / -11.60` |
| epsilon=0.10 | `-1.38 / -2.96` | `-5.90 / -7.08` |
| epsilon=0.20 | `+0.02 / -0.54` | `-2.82 / -4.36` |

Sinkhorn transport-plan 的边际残差约为 `1e-8`，说明运输计划满足数值边际约束；失败原因不是数值不守恒，而是 OT 梯度替代 CF 后改变了合成图更新方向，导致跨架构效用明显下降。增大 epsilon 只让 ConvNet 接近基线，没有修复 ResNet18 退化，因此没有授权继续跑 `NCFD + Sinkhorn` 剂量网格。


## 5. Energy 和 Sinkhorn 的区别

```text
Energy：
比较真实-合成、真实-真实、合成-合成的整体距离统计。

Sinkhorn：
显式计算 plan[i,j]，回答“谁向谁运输多少质量”。
```

| 项目 | Energy | Sinkhorn |
|---|---|---|
| 是否有运输计划 | 没有 | 有 `plan[i,j]` |
| 主要对象 | 距离统计 | 成本矩阵和运输质量 |
| 关键参数 | `unbiased` | `epsilon`、`iterations` |
| 代码入口 | `energy_distance()` | `sinkhorn_plan()`、`debiased_sinkhorn_divergence()` |

## 6. `0.25%` 梯度剂量

配置：

```yaml
geometric_target_gradient_dose: 0.0025
```

因为：

$$
0.0025=0.25\%
$$

它不是：

```text
geometric_loss / total_loss = 0.25%
```

它控制的是加权梯度：

$$
\frac{\|\nabla_\theta(wL_{geo})\|}
{\|\nabla_\theta L_{global}\|}
\approx0.0025
$$

例子：

```text
主损失梯度范数 = 100
几何损失原始梯度范数 = 40
```

原始比例：

$$
raw\_ratio=40/100=0.4
$$

目标比例为 `0.0025`，所以：

$$
w=0.0025/0.4=0.00625
$$

加权后的几何梯度：

$$
0.00625\times40=0.25
$$

最终比例：

$$
0.25/100=0.0025=0.25\%
$$

代码：

```python
raw_ratio = auxiliary_norm / global_norm
weight = target / raw_ratio
total_loss = global_loss + weight * geometric_loss
```

所以：

```text
0.25% 控制几何项对图像更新的影响力，
不是控制几何 loss 数值占 total loss 的比例。
```

## 7. NCFM 中的完整流程

```text
真实图像 img_real
        │
        ▼
模型提取 real_features
        │
        ├── Energy：三类距离统计
        ├── Sinkhorn：运输计划
        └── SW：多个方向的一维 W1
                        │
合成图像 img_syn ───────┘
                        ▼
                 geometric_loss
                        │
                        ▼
              测量几何项/主损失梯度比例
                        │
                        ▼
              自动得到 geometric_weight
                        │
                        ▼
       global_loss + weight * geometric_loss
                        │
                        ▼
                    更新合成图像
```

统一入口：

```python
geometric_loss, diagnostics = geometric_feature_loss(
    img_real,
    img_syn,
    model,
    args,
)
```

选择方法：

```yaml
geometric_method: energy
```

或：

```yaml
geometric_method: sinkhorn
```

## 8. 源码位置

```text
Energy：
NCFM_DISTANCE_BENCHMARK_20260727/energy/loss.py

Sinkhorn：
NCFM_DISTANCE_BENCHMARK_20260727/sinkhorn/loss.py

统一训练接口：
NCFM_DISTANCE_EXECUTION_20260727/recovery_v6_source/code_controlled_v6/NCFM/geometric_distance.py

训练接线：
NCFM_DISTANCE_EXECUTION_20260727/recovery_v6_source/code_controlled_v6/condenser/compute_loss.py
```

