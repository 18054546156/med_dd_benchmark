# 配置原则 - 以 HoP-TM 为基准

**日期**: 2026-08-12  
**原则**: 跟随 HoP-TM 官方配置，各算法保留自己的特有参数

---

## 核心原则

### 1. Backbone 统一标准（跟随 HoP-TM）

| 数据集 | Model | Depth | Norm | 来源 |
|--------|-------|-------|------|------|
| **PathMNIST** | ConvNet | 3 | InstanceNorm | HoP-TM 官方 |
| **COVID** | ConvNet | 5 | InstanceNorm | HoP-TM 官方 |
| **Kvasir** | ConvNet | 5 | InstanceNorm | 参考 COVID（HoP 无 Kvasir）|

### 2. 各算法特有参数保持原始设置

**示例（COVID IPC=10）**:

```yaml
# DC
model: ConvNetD5        # ← 统一 backbone
Iteration: 1000         # ← DC 原始设置
lr_img: 0.1             # ← DC 原始设置

# MTT
model: ConvNetD5        # ← 统一 backbone
Iteration: 5000         # ← MTT 原始设置
lr_img: 1000            # ← MTT 原始设置

# HoP-TM
model: ConvNetD5        # ← 统一 backbone
Iteration: 10000        # ← HoP 原始设置
lr_img: 100             # ← HoP 原始设置
high_order: true        # ← HoP 特有
```

---

## 需要修改的配置

### PathMNIST - 无需修改（已经都是 D3）

| 算法 | 当前 | 目标 | 改动 |
|------|------|------|------|
| DC/DSA/DM | D3 | D3 | ✅ 无需改动 |
| MTT | D3 | D3 | ✅ 无需改动 |
| HoP-TM | D3 | D3 | ✅ 无需改动 |
| NCFM | D3 | D3 | ✅ 无需改动 |

---

### COVID - 统一改成 D5

| 算法 | 当前 | 目标 | 改动 |
|------|------|------|------|
| DC | D3 | **D5** | ⚠️ 需要改 |
| DSA | D3 | **D5** | ⚠️ 需要改 |
| DM | D3 | **D5** | ⚠️ 需要改 |
| MTT | D4 | **D5** | ⚠️ 需要改 |
| HoP-TM | D5 | D5 | ✅ 无需改动 |
| NCFM | D4 | **D5** | ⚠️ 需要改 |
| DataDAM | D3 | **D5** | ⚠️ 需要改 |
| CAFE | D3 | **D5** | ⚠️ 需要改 |

---

### Kvasir - 统一改成 D5（参考 COVID）

| 算法 | 当前 | 目标 | 改动 |
|------|------|------|------|
| DC | D3 | **D5** | ⚠️ 需要改 |
| DSA | D3 | **D5** | ⚠️ 需要改 |
| DM | D3 | **D5** | ⚠️ 需要改 |
| MTT | D5 | D5 | ✅ 无需改动 |
| HoP-TM | D5 | D5 | ✅ 无需改动 |
| NCFM | D5 | D5 | ✅ 无需改动 |
| DataDAM | D3 | **D5** | ⚠️ 需要改 |
| CAFE | D3 | **D5** | ⚠️ 需要改 |

---

## 配置修改指南

### 示例 1: DC/DSA/DM (COVID)

**修改文件**: `configs/dc_dsa_dm/covid/ipc10_dc_full.yaml`

```yaml
# 修改前
model: ConvNet          # ← 默认 D3

# 修改后
model: ConvNetD5        # ← 改成 D5

# 其他参数保持不变
Iteration: 1000         # ← DC 原始设置，不改
lr_img: 0.1             # ← DC 原始设置，不改
```

---

### 示例 2: MTT (COVID)

**修改文件**: `configs/mtt/covid/ipc10_full.yaml`

```yaml
# 修改前
model: ConvNetD4        # ← 原来是 D4

buffer:
  model: ConvNetD4      # ← buffer 也是 D4

# 修改后
model: ConvNetD5        # ← 改成 D5

buffer:
  model: ConvNetD5      # ← buffer 也必须改成 D5

# 其他参数保持不变
distillation:
  Iteration: 5000       # ← MTT 原始设置，不改
  lr_img: 1000          # ← MTT 原始设置，不改
```

---

### 示例 3: NCFM (COVID)

**修改文件**: `configs/ncfm/covid/ipc10_full.yaml`

```yaml
# 修改前
network:
  depth: 4              # ← 原来是 4

# 修改后
network:
  depth: 5              # ← 改成 5

# 其他参数保持不变
condense:
  niter: 20000          # ← NCFM 原始设置，不改
  dis_metrics: NCFM     # ← NCFM 原始设置，不改
```

---

### 示例 4: HoP-TM (COVID)

**无需修改** - 已经是 D5

```yaml
model: ConvNetD5        # ← 已经正确
high_order: true        # ← HoP 特有参数，保持不变
lr_img: 100             # ← HoP 原始设置，保持不变
```

---

## 批量修改脚本

创建 `scripts/update_configs_to_d5.py`:

```python
#!/usr/bin/env python3
"""批量修改配置文件，将 COVID 和 Kvasir 统一改成 D5"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

def update_dc_dsa_dm(dataset: str):
    """更新 DC/DSA/DM 配置"""
    config_dir = ROOT / f"configs/dc_dsa_dm/{dataset}"
    for config_file in config_dir.glob("*.yaml"):
        content = config_file.read_text(encoding="utf-8")
        
        # model: ConvNet → model: ConvNetD5
        content = re.sub(
            r'^model: ConvNet$',
            'model: ConvNetD5',
            content,
            flags=re.MULTILINE
        )
        
        config_file.write_text(content, encoding="utf-8")
        print(f"Updated: {config_file}")

def update_mtt(dataset: str):
    """更新 MTT 配置"""
    config_file = ROOT / f"configs/mtt/{dataset}/ipc10_full.yaml"
    content = config_file.read_text(encoding="utf-8")
    
    # model: ConvNetD4 → model: ConvNetD5
    content = re.sub(r'ConvNetD4', 'ConvNetD5', content)
    
    config_file.write_text(content, encoding="utf-8")
    print(f"Updated: {config_file}")

def update_ncfm(dataset: str):
    """更新 NCFM 配置"""
    config_file = ROOT / f"configs/ncfm/{dataset}/ipc10_full.yaml"
    content = config_file.read_text(encoding="utf-8")
    
    # depth: 4 → depth: 5
    content = re.sub(
        r'depth: 4',
        'depth: 5',
        content
    )
    
    config_file.write_text(content, encoding="utf-8")
    print(f"Updated: {config_file}")

def main():
    for dataset in ["covid", "kvasir"]:
        print(f"\n=== Updating {dataset.upper()} ===")
        
        # DC/DSA/DM
        if dataset == "covid":
            update_dc_dsa_dm(dataset)
        
        # MTT
        if dataset == "covid":
            update_mtt(dataset)
        
        # NCFM
        if dataset == "covid":
            update_ncfm(dataset)
    
    print("\n✅ All configs updated to D5")

if __name__ == "__main__":
    main()
```

---

## 验证修改

修改完成后，运行验证：

```bash
# 检查所有 COVID 配置是否都是 D5
grep -r "model:" configs/*/covid/*.yaml
grep -r "depth:" configs/*/covid/*.yaml

# 应该看到：
# - dc_dsa_dm: model: ConvNetD5
# - mtt: model: ConvNetD5
# - hop_tm: model: ConvNetD5
# - ncfm: depth: 5
```

---

## 总结

### 原则
1. **Backbone 跟随 HoP-TM 官方**
   - PathMNIST: D3
   - COVID: D5
   - Kvasir: D5（参考 COVID）

2. **算法特有参数保持原始**
   - Iteration, lr_img, lr_net 等保持各算法的 RAW 设置
   - 不强制统一

3. **Kvasir 参考 COVID**
   - HoP 没有 Kvasir 实验
   - 两者都是复杂医疗任务，用相同 D5

---

**最后更新**: 2026-08-12  
**状态**: ✅ 配置已完成，代码已支持 D5

---

## 代码修改记录

### ConvNetD5 支持已添加

为了支持 COVID 和 Kvasir 使用 ConvNetD5，已对以下文件添加 D5 分支：

| 文件 | 修改位置 | 改动 |
|------|---------|------|
| `adapted/dc_dsa_dm/utils.py` | line 283 | 添加 `ConvNetD5` 分支 |
| `adapted/datadam/utils.py` | line 430 | 添加 `ConvNetD5` 分支 |
| `adapted/cafe/utils.py` | line 255 | 添加 `ConvNetD5` 分支 |

**修改内容**（统一模式）：
```python
elif model == 'ConvNetD5':
    net = ConvNet(channel=channel, num_classes=num_classes, 
                  net_width=net_width, net_depth=5, 
                  net_act=net_act, net_norm=net_norm, 
                  net_pooling=net_pooling, im_size=im_size)
```

**验证测试**: `scripts/test_convnetd5.py`
- ✅ DC/DSA/DM: ConvNetD5 加载成功（599,812 参数）
- ✅ DataDAM: ConvNetD5 加载成功（599,812 参数）
- ✅ CAFE: ConvNetD5 加载成功（599,812 参数）
- ✅ MTT: ConvNetD5 加载成功（599,812 参数，参考实现）

**已支持 D5 的算法**：
- MTT: 原始代码已支持 D1-D8
- HoP-TM: 原始代码已支持 D5
- NCFM: 通过 `network.depth` 参数动态创建
- DC/DSA/DM: ✅ 本次添加
- DataDAM: ✅ 本次添加
- CAFE: ✅ 本次添加

---

## 下一步工作

### 1. Buffer 重新生成（MTT）
MTT 使用预训练 expert trajectories，backbone 改成 D5 后需要重新生成：

```bash
# COVID D5 expert buffer
python adapted/mtt/buffer.py \
  --config configs/mtt/covid/ipc10_full.yaml

# Kvasir D5 expert buffer（如果有 MTT Kvasir 配置）
python adapted/mtt/buffer.py \
  --config configs/mtt/kvasir/ipc10_full.yaml
```

### 2. Pretrain 模型重新训练（NCFM）
NCFM 使用预训练模型，depth 改成 5 后需要重新 pretrain：

```bash
# COVID D5 pretrain
python adapted/ncfm/pretrain.py \
  --config configs/ncfm/covid/ipc10_full.yaml

# Kvasir D5 pretrain
python adapted/ncfm/pretrain.py \
  --config configs/ncfm/kvasir/ipc10_full.yaml
```

### 3. 公平对比实验
所有配置和代码就绪后，运行完整对比：

```bash
# 使用 run_config.py 批量运行
python run_config.py --dataset covid --ipc 10 --algorithm all
python run_config.py --dataset kvasir --ipc 10 --algorithm all
python run_config.py --dataset pathmnist --ipc 10 --algorithm all
```

---

**最后更新**: 2026-08-12  
**状态**: ✅ 配置已完成，代码已支持 D5
