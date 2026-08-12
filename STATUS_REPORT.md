# 公平对比配置完成报告（修正版）

**日期**: 2026-08-12  
**状态**: ✅ 代码修复完成，真实测试全部通过

---

## 完成情况总览

### ✅ 已完成
1. **代码支持 ConvNetD5**
   - DC/DSA/DM、DataDAM、CAFE 添加 D5 分支
   - CAFE 修复 im_size 缺失问题（D1-D5 全部修复）
   - 所有算法通过真实前向测试（112×112 和 128×128）

2. **配置文件统一**
   - PathMNIST: 9个配置，全部 ConvNet (D3)
   - COVID: 11个配置，全部 ConvNetD5
   - Kvasir: 11个配置，全部 ConvNetD5
   - MTT 配置冲突已修复（depth 4→5）

3. **测试验证完成**
   - 重写测试脚本：subprocess 隔离 + 真实前向
   - 测试 112×112 和 128×128 两种输入
   - 所有算法测试通过 ✅

4. **文档更新**
   - `CONFIG_PRINCIPLES.md`: 配置原则和批量修改指南
   - `CONVNETD5_SUPPORT.md`: 问题发现与修复详细报告
   - `STATUS_REPORT.md`: 本报告

---

## 问题发现与修复

### P0-1: CAFE im_size 缺失 ✅
**问题**: ConvNetD1-D5 没有传递 im_size，医疗数据集前向失败  
**修复**: 为所有 depth 分支添加 `im_size=im_size`  
**验证**: 112×112 和 128×128 前向测试通过

### P0-2: 测试脚本不可靠 ✅
**问题**: 模块缓存、没有真实前向、硬编码路径  
**修复**: 重写为 subprocess 隔离 + 真实前向传播  
**验证**: 8 个测试用例（4 算法 × 2 尺寸）全部通过

### P1: MTT 配置冲突 ✅
**问题**: `model: ConvNetD5` 但 `network.depth: 4`  
**修复**: 修改为 `network.depth: 5`  
**影响**: COVID 配置已修复，Kvasir 本来就正确

---

## 真实测试结果

### 测试环境
- **方法**: 独立子进程，CPU 模式
- **输入**: batch=2, channel=3, classes=4
- **尺寸**: 112×112 (COVID) 和 128×128 (Kvasir)

### 测试结果 ✅ 全部通过

| 算法 | 112×112 参数量 | 112×112 前向 | 128×128 参数量 | 128×128 前向 |
|------|---------------|-------------|---------------|-------------|
| DC/DSA/DM | 599,812 | ✅ | 603,396 | ✅ |
| DataDAM | 599,812 | ✅ | 603,396 | ✅ |
| CAFE | 599,812 | ✅ | 603,396 | ✅ |
| MTT | 599,812 | ✅ | 603,396 | ✅ |

所有算法输出形状正确：`torch.Size([2, 4])`

---

## 配置统计

### PathMNIST（ConvNet D3）
| 算法 | 配置数 | Model | 验证 |
|------|--------|-------|------|
| DC/DSA/DM | 3 | ConvNet | ✅ |
| MTT | 1 | ConvNet | ✅ |
| HoP-TM | 2 | ConvNet | ✅ |
| NCFM | 2 | depth: 3 | ✅ |
| DataDAM | 1 | ConvNet | ✅ |

**总计**: 9个配置 ✅

---

### COVID（ConvNetD5）
| 算法 | 配置数 | Model | 前向测试 |
|------|--------|-------|---------|
| DC/DSA/DM | 3 | ConvNetD5 | ✅ |
| MTT | 1 | ConvNetD5 | ✅ |
| HoP-TM | 2 | ConvNetD5 | - |
| NCFM | 2 | depth: 5 | - |
| DataDAM | 1 | ConvNetD5 | ✅ |
| CAFE | 1 | ConvNetD5 | ✅ |

**总计**: 10个配置 ✅（含 smoke test）

---

### Kvasir（ConvNetD5）
| 算法 | 配置数 | Model | 前向测试 |
|------|--------|-------|---------|
| DC/DSA/DM | 3 | ConvNetD5 | ✅ |
| MTT | 1 | ConvNetD5 | ✅ |
| HoP-TM | 2 | ConvNetD5 | - |
| NCFM | 2 | depth: 5 | - |
| DataDAM | 1 | ConvNetD5 | ✅ |
| CAFE | 1 | ConvNetD5 | ✅ |

**总计**: 10个配置 ✅（含 smoke test）

---

## 算法支持状态

### 可以立即运行（不需要预生成资源）

| 算法 | D5 代码 | 前向测试 | 状态 |
|------|---------|---------|------|
| DC | ✅ | ✅ | ✅ 就绪 |
| DSA | ✅ | ✅ | ✅ 就绪 |
| DM | ✅ | ✅ | ✅ 就绪 |
| DataDAM | ✅ | ✅ | ✅ 就绪 |
| CAFE | ✅ | ✅ | ✅ 就绪 |

### 需要预生成资源（GPU）

| 算法 | D5 代码 | 前向测试 | 需要生成 | 状态 |
|------|---------|---------|---------|------|
| MTT | ✅ | ✅ | Expert buffer | ⏳ 等待 GPU |
| HoP-TM | ✅ | - | Expert buffer | ⏳ 等待 GPU |
| NCFM | ✅ | - | Pretrain 模型 | ⏳ 等待 GPU |

---

## 代码修改清单

### 新增 ConvNetD5 支持

| 文件 | 行号 | 修改 |
|------|------|------|
| `adapted/dc_dsa_dm/utils.py` | 283 | 添加 ConvNetD5 分支 |
| `adapted/datadam/utils.py` | 430 | 添加 ConvNetD5 分支 |
| `adapted/cafe/utils.py` | 255 | 添加 ConvNetD5 分支 |

### 修复 im_size 传递

| 文件 | 行号 | 修改 |
|------|------|------|
| `adapted/cafe/utils.py` | 247-256 | D1-D5 全部添加 im_size 参数 |

**修改内容**：
```python
# 所有 ConvNetD1-D5 分支统一添加
net = ConvNet(..., im_size=im_size)
```

---

## 配置修改清单

### 批量更新为 D5
- `configs/dc_dsa_dm/covid/*.yaml`: ConvNet → ConvNetD5
- `configs/dc_dsa_dm/kvasir/*.yaml`: ConvNet → ConvNetD5
- `configs/datadam/covid/*.yaml`: ConvNet → ConvNetD5
- `configs/datadam/kvasir/*.yaml`: ConvNet → ConvNetD5
- `configs/cafe/covid/*.yaml`: ConvNet → ConvNetD5
- `configs/cafe/kvasir/*.yaml`: ConvNet → ConvNetD5
- `configs/mtt/covid/*.yaml`: ConvNetD4 → ConvNetD5
- `configs/ncfm/covid/*.yaml`: depth: 4 → depth: 5

### 修复配置冲突
- `configs/mtt/covid/ipc10_full.yaml`: network.depth: 4 → 5

---

## 实验运行指南

### 阶段 1: 快速验证（立即可运行）

```bash
# COVID - 不需要 buffer 的算法
python run_config.py configs/dc_dsa_dm/covid/ipc10_dc_full.yaml
python run_config.py configs/dc_dsa_dm/covid/ipc10_dsa_full.yaml
python run_config.py configs/dc_dsa_dm/covid/ipc10_dm_full.yaml
python run_config.py configs/datadam/covid/ipc10_full.yaml
python run_config.py configs/cafe/covid/ipc10_full.yaml

# Kvasir - 不需要 buffer 的算法
python run_config.py configs/dc_dsa_dm/kvasir/ipc10_dc_full.yaml
python run_config.py configs/dc_dsa_dm/kvasir/ipc10_dsa_full.yaml
python run_config.py configs/dc_dsa_dm/kvasir/ipc10_dm_full.yaml
python run_config.py configs/datadam/kvasir/ipc10_full.yaml
python run_config.py configs/cafe/kvasir/ipc10_full.yaml
```

### 阶段 2: 生成预训练资源（需要 GPU）

```bash
# MTT expert buffers (D5)
python adapted/mtt/buffer.py --config configs/mtt/covid/ipc10_full.yaml
python adapted/mtt/buffer.py --config configs/mtt/kvasir/ipc10_full.yaml

# HoP-TM expert buffers (D5)
python adapted/hop_tm/buffer.py --config configs/hop_tm/covid/ipc10_full.yaml
python adapted/hop_tm/buffer.py --config configs/hop_tm/kvasir/ipc10_full.yaml

# NCFM pretrain (depth=5)
python adapted/ncfm/pretrain.py --config configs/ncfm/covid/ipc10_full.yaml
python adapted/ncfm/pretrain.py --config configs/ncfm/kvasir/ipc10_full.yaml
```

### 阶段 3: 完整实验（资源生成后）

```bash
# 运行需要 buffer 的算法
python run_config.py configs/mtt/covid/ipc10_full.yaml
python run_config.py configs/mtt/kvasir/ipc10_full.yaml
python run_config.py configs/hop_tm/covid/ipc10_full.yaml
python run_config.py configs/hop_tm/kvasir/ipc10_full.yaml
python run_config.py configs/ncfm/covid/ipc10_full.yaml
python run_config.py configs/ncfm/kvasir/ipc10_full.yaml
```

---

## 网络输出格式

不同算法的 `forward()` 返回格式：

| 算法 | 返回格式 | 测试脚本处理 |
|------|---------|-------------|
| DC/DSA/DM | `logits: Tensor` | 直接使用 |
| MTT | `logits: Tensor` | 直接使用 |
| DataDAM | `(features, logits): tuple` | 取 tuple[1] |
| CAFE | `(logits, [features]): tuple` | 取 tuple[0] |

测试脚本已正确处理所有格式。

---

## 参数量统计

| 数据集 | 输入尺寸 | ConvNetD5 参数量 |
|--------|---------|-----------------|
| COVID | 112×112 | 599,812 |
| Kvasir | 128×128 | 603,396 |
| PathMNIST | 32×32 (pad) | ~597,252 |

**差异原因**: 分类器输入维度 = 特征图尺寸 × channel_width

---

## 文件清单

### 新增文件
- `scripts/update_configs_to_d5.py` - 批量配置更新脚本
- `scripts/test_convnetd5.py` - 真实前向测试脚本
- `CONFIG_PRINCIPLES.md` - 配置原则文档
- `CONVNETD5_SUPPORT.md` - 问题发现与修复详细报告
- `STATUS_REPORT.md` - 本报告

### 修改文件（代码）
- `adapted/dc_dsa_dm/utils.py` - 添加 D5 支持
- `adapted/datadam/utils.py` - 添加 D5 支持
- `adapted/cafe/utils.py` - 添加 D5 支持 + 修复 im_size

### 修改文件（配置）
- `configs/*/covid/*.yaml` - 更新为 D5
- `configs/*/kvasir/*.yaml` - 更新为 D5
- `configs/mtt/covid/ipc10_full.yaml` - 修复 depth 冲突

---

## 验证清单

### 代码修改
- [x] DC/DSA/DM 添加 ConvNetD5 分支
- [x] DataDAM 添加 ConvNetD5 分支
- [x] CAFE 添加 ConvNetD5 分支
- [x] CAFE 修复 im_size 传递（D1-D5）
- [x] Python 语法检查通过

### 真实测试
- [x] DC/DSA/DM: 112×112 前向通过
- [x] DC/DSA/DM: 128×128 前向通过
- [x] DataDAM: 112×112 前向通过
- [x] DataDAM: 128×128 前向通过
- [x] CAFE: 112×112 前向通过
- [x] CAFE: 128×128 前向通过
- [x] MTT: 112×112 前向通过
- [x] MTT: 128×128 前向通过

### 配置修改
- [x] COVID 配置全部更新为 D5
- [x] Kvasir 配置全部更新为 D5
- [x] MTT 配置冲突已修复
- [x] PathMNIST 保持 D3

### 待办事项
- [ ] 同步到服务器（等待 VPN）
- [ ] 生成 MTT D5 buffers（需要 GPU）
- [ ] 生成 HoP-TM D5 buffers（需要 GPU）
- [ ] 训练 NCFM depth=5 pretrain（需要 GPU）
- [ ] 运行完整公平对比实验

---

## 技术要点

### 为什么必须修改代码？

**错误做法**（只改 YAML）：
```yaml
model: ConvNetD5  # 配置写 D5
```
运行时报错：`unknown model: ConvNetD5`

**正确做法**（配置 + 代码）：
```yaml
model: ConvNetD5  # 配置写 D5
```
```python
elif model == 'ConvNetD5':  # 代码支持 D5
    net = ConvNet(..., net_depth=5, ...)
```

### 为什么 CAFE 需要 im_size？

ConvNet 的分类器维度由特征图尺寸决定：
```python
features_out_shape = (C, H_pool, W_pool)
classifier_in = C * H_pool * W_pool  # 必须知道 im_size 才能计算
```

如果不传 im_size，默认 32×32 → 112×112 输入时维度不匹配。

### 为什么参数量不完全相同？

虽然都是 ConvNetD5，但分类器的输入维度随 pooling 后特征图尺寸变化：
- 112×112 → pooling 后某尺寸 → 599,812 参数
- 128×128 → pooling 后某尺寸 → 603,396 参数

卷积层参数相同，差异在最后的全连接层。

---

**最后更新**: 2026-08-12  
**完成者**: Claude Opus 5  
**状态**: ✅ 代码修复完成，真实测试全部通过，等待 VPN 恢复后进行实验
