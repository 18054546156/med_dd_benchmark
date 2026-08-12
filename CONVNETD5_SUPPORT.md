# ConvNetD5 支持实现报告（修正版）

**日期**: 2026-08-12  
**状态**: ✅ 代码修复完成，测试全部通过

---

## 问题发现与修复

### P0-1: CAFE 的 im_size 缺失 ✅ 已修复

**问题**：CAFE 的 ConvNetD1-D5 分支都没有传递 `im_size` 参数，导致医疗数据集（112×112、128×128）前向传播时维度不匹配。

**位置**：`adapted/cafe/utils.py:247-256`

**修复**：为 ConvNetD1-D5 所有分支添加 `im_size=im_size` 参数：

```python
# 修复前
elif model == 'ConvNetD5':
    net = ConvNet(channel=channel, num_classes=num_classes, 
                  net_width=net_width, net_depth=5, 
                  net_act=net_act, net_norm=net_norm, 
                  net_pooling=net_pooling)  # ← 缺少 im_size

# 修复后
elif model == 'ConvNetD5':
    net = ConvNet(channel=channel, num_classes=num_classes, 
                  net_width=net_width, net_depth=5, 
                  net_act=net_act, net_norm=net_norm, 
                  net_pooling=net_pooling, 
                  im_size=im_size)  # ← 添加 im_size
```

**影响**：D1-D5 全部修复，保持行为一致。

---

### P0-2: 测试脚本不可靠 ✅ 已修复

**问题 1**：Python 模块缓存导致所有算法实际使用同一个 `utils.py`

**问题 2**：只测试模型创建，没有执行真实前向传播

**问题 3**：硬编码项目路径

**修复**：重写测试脚本 `scripts/test_convnetd5.py`：
- 使用 subprocess 独立进程测试每个算法
- 执行真实前向传播（batch=2, 112×112 和 128×128）
- 使用 `Path(__file__).resolve()` 定位项目根目录
- 处理不同网络输出格式（tensor、tuple、list）

---

### P1: MTT 配置冲突 ✅ 已修复

**问题**：`configs/mtt/covid/ipc10_full.yaml` 顶层 `model: ConvNetD5`，但 `network.depth: 4` 冲突。

**修复**：
```yaml
# 修复前
model: ConvNetD5
network:
  depth: 4  # ← 不一致

# 修复后
model: ConvNetD5
network:
  depth: 5  # ← 一致
```

---

## 最终测试结果

### 测试配置
- **算法**：DC/DSA/DM, DataDAM, CAFE, MTT
- **输入尺寸**：112×112 (COVID), 128×128 (Kvasir)
- **批次大小**：2
- **类别数**：4
- **模式**：CPU（避免 CUDA 环境差异）

### 测试结果 ✅ 全部通过

```
======================================================================
Testing ConvNetD5 with Real Forward Pass
======================================================================

[TEST] DC/DSA/DM
  Testing 112x112...
    [OK] Model created: 599812 parameters
    [OK] Forward pass succeeded, output shape torch.Size([2, 4])
  Testing 128x128...
    [OK] Model created: 603396 parameters
    [OK] Forward pass succeeded, output shape torch.Size([2, 4])

[TEST] DataDAM
  Testing 112x112...
    [OK] Model created: 599812 parameters
    [OK] Forward pass succeeded, output shape torch.Size([2, 4])
  Testing 128x128...
    [OK] Model created: 603396 parameters
    [OK] Forward pass succeeded, output shape torch.Size([2, 4])

[TEST] CAFE
  Testing 112x112...
    [OK] Model created: 599812 parameters
    [OK] Forward pass succeeded, output shape torch.Size([2, 4])
  Testing 128x128...
    [OK] Model created: 603396 parameters
    [OK] Forward pass succeeded, output shape torch.Size([2, 4])

[TEST] MTT
  Testing 112x112...
    [OK] Model created: 599812 parameters
    [OK] Forward pass succeeded, output shape torch.Size([2, 4])
  Testing 128x128...
    [OK] Model created: 603396 parameters
    [OK] Forward pass succeeded, output shape torch.Size([2, 4])

======================================================================
SUMMARY
======================================================================
[OK] DC/DSA/DM
[OK] DataDAM
[OK] CAFE
[OK] MTT
======================================================================
All tests passed!
```

---

## 修改汇总

### 代码修改

| 文件 | 行号 | 修改内容 |
|------|------|----------|
| `adapted/dc_dsa_dm/utils.py` | 283 | 添加 ConvNetD5 分支 |
| `adapted/datadam/utils.py` | 430 | 添加 ConvNetD5 分支 |
| `adapted/cafe/utils.py` | 247-256 | 添加 ConvNetD5 分支 + 修复 D1-D5 im_size |

### 配置修改

| 文件 | 修改 | 说明 |
|------|------|------|
| `configs/mtt/covid/ipc10_full.yaml` | depth: 4 → 5 | 修复与 model: ConvNetD5 的冲突 |
| `configs/*/covid/*.yaml` | model → ConvNetD5 | 统一 COVID 使用 D5 |
| `configs/*/kvasir/*.yaml` | model → ConvNetD5 | 统一 Kvasir 使用 D5 |

### 测试脚本

| 文件 | 说明 |
|------|------|
| `scripts/test_convnetd5.py` | 完全重写：subprocess 隔离 + 真实前向 |

---

## 算法支持状态（最终）

| 算法 | ConvNetD5 代码 | 真实前向 112×112 | 真实前向 128×128 | 状态 |
|------|---------------|-----------------|-----------------|------|
| **DC/DSA/DM** | ✅ 新增 | ✅ 通过 | ✅ 通过 | ✅ 就绪 |
| **DataDAM** | ✅ 新增 | ✅ 通过 | ✅ 通过 | ✅ 就绪 |
| **CAFE** | ✅ 新增+修复 | ✅ 通过 | ✅ 通过 | ✅ 就绪 |
| **MTT** | ✅ 原本支持 | ✅ 通过 | ✅ 通过 | ⏳ 需重新生成 buffer |
| **HoP-TM** | ✅ 原本支持 | - | - | ⏳ 需重新生成 buffer |
| **NCFM** | ✅ 原本支持 | - | - | ⏳ 需重新 pretrain |

**说明**：
- ✅ 就绪：代码和配置完成，可直接运行蒸馏
- ⏳ 需准备：需要 GPU 预生成 expert buffer 或 pretrain 模型

---

## 网络输出格式差异

不同算法的网络 `forward()` 返回格式不同：

| 算法 | 返回格式 | 说明 |
|------|---------|------|
| DC/DSA/DM | `logits` | 标准分类输出 |
| MTT | `logits` | 标准分类输出 |
| DataDAM | `(features, logits)` | 元组：(embedding, 分类) |
| CAFE | `(logits, [features])` | 元组：(分类, 特征列表) |

测试脚本已正确处理所有格式。

---

## 参数量对比

| 输入尺寸 | ConvNetD5 参数量 | 备注 |
|---------|-----------------|------|
| 112×112 | 599,812 | COVID |
| 128×128 | 603,396 | Kvasir |
| 32×32 | 597,252 | CIFAR10（参考） |

**差异原因**：分类器输入维度随 pooling 后的特征图尺寸变化。

---

## 下一步工作

### 1. 生成预训练资源（需要 GPU 和 VPN）

```bash
# MTT expert buffers
python adapted/mtt/buffer.py --config configs/mtt/covid/ipc10_full.yaml
python adapted/mtt/buffer.py --config configs/mtt/kvasir/ipc10_full.yaml

# HoP-TM expert buffers
python adapted/hop_tm/buffer.py --config configs/hop_tm/covid/ipc10_full.yaml
python adapted/hop_tm/buffer.py --config configs/hop_tm/kvasir/ipc10_full.yaml

# NCFM pretrain
python adapted/ncfm/pretrain.py --config configs/ncfm/covid/ipc10_full.yaml
python adapted/ncfm/pretrain.py --config configs/ncfm/kvasir/ipc10_full.yaml
```

### 2. 运行公平对比实验

```bash
# 可以立即运行的算法（不需要 buffer）
python run_config.py configs/dc_dsa_dm/covid/ipc10_dc_full.yaml
python run_config.py configs/dc_dsa_dm/covid/ipc10_dsa_full.yaml
python run_config.py configs/dc_dsa_dm/covid/ipc10_dm_full.yaml
python run_config.py configs/datadam/covid/ipc10_full.yaml
python run_config.py configs/cafe/covid/ipc10_full.yaml

# Buffer 生成后运行
python run_config.py configs/mtt/covid/ipc10_full.yaml
python run_config.py configs/hop_tm/covid/ipc10_full.yaml

# Pretrain 完成后运行
python run_config.py configs/ncfm/covid/ipc10_full.yaml
```

---

## 验证清单

- [x] DC/DSA/DM 代码添加 D5 分支
- [x] DataDAM 代码添加 D5 分支
- [x] CAFE 代码添加 D5 分支 + 修复 im_size
- [x] MTT 配置修复 depth 冲突
- [x] 真实前向测试 112×112 全部通过
- [x] 真实前向测试 128×128 全部通过
- [x] 配置文件统一为 D5
- [ ] MTT D5 expert buffer 生成（需要 GPU）
- [ ] HoP-TM D5 expert buffer 生成（需要 GPU）
- [ ] NCFM depth=5 pretrain（需要 GPU）
- [ ] 完整实验运行（需要服务器）

---

**最后更新**: 2026-08-12  
**状态**: ✅ 代码和配置修复完成，真实测试全部通过
