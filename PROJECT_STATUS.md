# 项目当前状态与待解决问题

**最后更新**: 2026-08-13  
**状态**: 医疗数据适配层完成，但不是生产级Benchmark

---

## ✅ 已完成部分

### 1. 数据准备
- ✅ 统一医疗数据接口 (`utils/medical_dataset_utils.py`)
- ✅ 三个数据集的train/val/test分割：
  - PathMNIST: 89996 / 10004 / 7180
  - COVID: 14817 / 2116 / 4232
  - Kvasir: 5600 / 800 / 1600
- ✅ 标签标量化和尺寸合同

### 2. 算法适配
- ✅ `adapted/` 中加入医疗数据加载
- ✅ 支持ConvNetD5（PathMNIST D3, COVID/Kvasir D5）
- ✅ 25个配置文件，配置静态校验通过（0错误）
- ✅ Python语法检查通过（compileall）

### 3. 运行框架
- ✅ `scripts/run_config.py` 统一运行器
  - 阶段检查
  - Buffer存在性检查
  - NCFM合成数据路径解析
  - 运行manifest记录
- ✅ ArgsProcessor配置同名键冲突检测
- ✅ 统一评估脚本 (`scripts/unified_eval.py`)

### 4. 文档
- ✅ 算法完整指南 (`ALGORITHM_COMPLETE_GUIDE.md`)
- ✅ 原始函数对照表 (`RAW_FUNCTIONS_REFERENCE.md`)
- ✅ 修改分析 (`CHANGES_ANALYSIS.md`)
- ✅ HoP-TM/NCFM流程 (`HOP_NCFM_PIPELINE.md`)

### 5. Git管理
- ✅ `raw/` 保持原始代码，未检测到工作区修改
- ✅ `.gitignore` 排除 `raw/`, `data/`, `results/`, `buffers/`, `logs/`
- ✅ 外层Git不会推送原始仓库

---

## ❌ 关键阻塞问题

### 🔴 问题1: MTT/HoP-TM的D5 buffer缺失

**现状**:
```
当前buffer: buffers/.../COVID_NO_ZCA/ConvNet/
            buffers/.../Kvasir_NO_ZCA/ConvNet/

运行器查找: buffers/mtt/COVID_NO_ZCA/ConvNetD5/
            buffers/hop_tm/COVID_NO_ZCA/ConvNetD5/
```

**影响**: MTT/HoP-TM的COVID和Kvasir distill命令会失败

**检查位置**: 
- `run_config.py:376-384` (MTT)
- `run_config.py:411-419` (HoP-TM)

**解决方案**: 必须重新生成
```bash
MTT buffer:    PathMNIST D3, COVID D5, Kvasir D5
HoP-TM buffer: PathMNIST D3, COVID D5, Kvasir D5
```

**不能复用旧的ConvNet buffer**

---

### 🔴 问题2: NCFM pretrain数据重复Normalize

**现状**:
```
训练准确率: 84.3-84.8%
验证准确率: 52.0-53.1%  ← 异常低！
```

**根本原因**: 训练数据被重复Normalize
1. `medical_dataset_utils.py` 对PathMNIST做了一次Normalize
2. NCFM的`diffaug.py` 对训练输入又做了一次Normalize
3. 验证集只Normalize一次

**影响**: 训练/验证分布不一致，当前premodel0-4不能用于condense

**对比**: HoP buffer的Test Acc 88-90%，说明正常训练应该能达到这个水平

**解决方案**:
- NCFM pretrain的train split: 只做 `ToTensor + Resize`
- NCFM validation: 做 `ToTensor + Resize + Normalize`
- 保留NCFM的`diffaug`，由它负责训练时Normalize
- 当前`premodel0~4`建议归档后重新pretrain

**修改位置**: `utils/medical_dataset_utils.py` 或 `adapted/ncfm/utils/utils.py`

---

### 🟡 问题3: NCFM HPC脚本路径错误

**脚本检查路径** (`scripts/ncfm_pipeline.sbatch:36`):
```bash
pretrained_models/<dataset>/<dataset>/
```

**实际生成路径** (`adapted/ncfm/utils/init_script.py:103`):
```
pretrained_models/ncfm/
  COVID/
    ConvNetD5_IN_W128/
      seed0/
```

**影响**: "预训练已生成，但脚本检查不到"

**解决方案**: 修复HPC脚本的路径检查逻辑

---

### 🟡 问题4: backbone配置没有真正集中生效

**现状**:
- `configs/backbones/` 用于静态校验，不被`run_config.py`自动加载
- 各算法使用各自YAML中的`model`字段

**不能声称**: "所有算法强制使用统一backbone"

**实际情况**:
```
配置文件写 D5
校验器检查 D5
运行器使用各算法YAML中的model  ← 不是强制加载
```

**潜在问题**: 不同算法目录里的`ConvNetD5`实现可能存在：
- pooling差异
- normalization差异
- feature输出差异
- forward返回值差异

**不能仅凭名字宣称所有算法使用完全相同网络**

---

### 🟡 问题5: seed配置多数没有真正传入

**配置中写了**:
```yaml
seed: 0
```

**实际情况**:
- DC/DSA/DM/DataDAM/CAFE/MTT: 入口没有统一`--seed`传递机制
- HoP-TM: `distill_high_order_spl.py:28` 固定调用`manual_seed(0)`
- NCFM: seed为0时不会真正设置随机种子 (`init_script.py:177`)

**影响**: manifest中记录的seed不一定是实际使用的seed

**可复现性风险**: 无法通过配置文件控制随机种子

---

### 🟡 问题6: 实验输出目录没有真正统一

**logs目录** (统一):
```
logs/<algorithm>/<dataset>/<stage>/<timestamp>/
```

**results目录** (分散):
```
results/<algorithm>/<dataset>/ipc10/
results/ncfm/<dataset>/condense/...
logged_files/<dataset>/<wandb_id>/
```

**reorganize_results.py**: 明确写着自动迁移需要进一步实现 (`line 238`)

**影响**: 结果路径不统一，难以批量收集

---

### 🟡 问题7: 当前本机没有正式NCFM预训练目录

**不存在**:
```
pretrained_models/ncfm/
  COVID/ConvNetD5_IN_W128/seed0/  ← 正式版
  Kvasir/ConvNetD5_IN_W128/seed0/
```

**已有**: 主要是smoke或旧结果，不能作为当前D5正式配置的预训练资源

---

### 🟡 问题8: 当前结果大多是smoke/legacy

**非smoke的正式结果**: 主要只有DC的部分结果
```
results/dc_dsa_dm/<dataset>/DC/ipc10/
```

**不能声称**: "8个算法 × 3个数据集全部跑通"

**实际完成度**:
- 数据准备: ✅ 完成
- 医疗loader: ✅ 基本完成
- 配置静态校验: ✅ 完成
- DC/DSA/DM/DataDAM/CAFE: ⚠️ 入口可生成，完整训练未全部确认
- MTT: 🔴 buffer可生成命令，D5 distill被旧buffer阻塞
- HoP-TM: 🔴 buffer可生成命令，D5 distill被旧buffer阻塞
- NCFM: 🔴 需要先修复路径，并重新完成三个数据集pretrain
- 完整Benchmark: ❌ 尚未完成

---

### 🟢 问题9: 配置中文注释存在乱码

**示例** (`configs/ncfm/covid/ipc10_full.yaml:2`):
```yaml
# RAW 鍙傝€冿細
```

**影响**: 不影响运行，但影响审计和论文记录

**解决方案**: 统一转换为UTF-8并重新检查

---

### 🟢 问题10: GitHub克隆后不能直接运行

**被忽略的内容**:
```
raw/              ← 原始算法代码
data/*            ← 数据集
results/          ← 实验结果
buffers/          ← Expert trajectories
logs/             ← 运行日志
pretrained_models/ ← NCFM预训练模型
```

**影响**: 克隆后必须重新执行：
1. 数据准备
2. Buffer生成
3. NCFM pretrain

**这是正常的设计决策**: 避免Git仓库过大，但需要在README中说明

---

## 🎯 优先级修复顺序

### 🔴 P0 (阻塞实验)
1. **修复NCFM pretrain的重复Normalize问题**
   - 废弃当前premodel0-4
   - 修改数据加载逻辑
   - 重新运行PathMNIST/COVID/Kvasir的pretrain

2. **重新生成MTT/HoP-TM的D5 buffer**
   - PathMNIST: ConvNet (已有) 或 ConvNetD3
   - COVID: ConvNetD5 (重新生成)
   - Kvasir: ConvNetD5 (重新生成)

### 🟡 P1 (影响可靠性)
3. **修复NCFM HPC脚本路径检查**
4. **统一seed传递机制**
5. **验证ConvNetD5跨算法一致性**

### 🟢 P2 (完善性)
6. **修复配置中文乱码**
7. **统一实验输出目录结构**
8. **完成8算法×3数据集的正式实验**

---

## 📝 当前运行状态

### NCFM PathMNIST Pretrain (HPC)
- **作业**: 25039
- **节点**: hpcgpu102
- **状态**: RUNNING
- **已完成**: premodel0-4 (共5个模型)
- **正在进行**: premodel5 (第6个模型)
- **问题**: 训练数据重复Normalize，结果不可用
- **建议**: 停止作业，修复后重新运行

### HoP Buffer生成
- **参考结果**: Test Acc 88-90% (正常水平)
- **说明**: HoP teacher训练正常，可作为baseline

---

## 🔄 下一步行动

### 立即行动
1. **停止当前NCFM pretrain作业** (如果仍在运行)
2. **修复NCFM的Normalize问题**:
   - 选项A: 修改`medical_dataset_utils.py`，为NCFM提供不带Normalize的数据
   - 选项B: 修改`adapted/ncfm/utils/utils.py`，检测数据是否已Normalize
3. **重新启动NCFM pretrain** (3个数据集 × 20模型 × 60 epochs)

### 短期计划
4. **生成MTT D5 buffer** (COVID, Kvasir)
5. **生成HoP-TM D5 buffer** (COVID, Kvasir)
6. **验证DC/DSA/DM/DataDAM/CAFE在3个数据集上的完整训练**

### 中期计划
7. **运行完整Benchmark**: 8算法 × 3数据集 × IPC10
8. **使用unified_eval.py进行公平评估**
9. **整理结果和生成对比表**

---

## 📚 相关文档

- [ALGORITHM_COMPLETE_GUIDE.md](ALGORITHM_COMPLETE_GUIDE.md) - 算法完整指南
- [RAW_FUNCTIONS_REFERENCE.md](RAW_FUNCTIONS_REFERENCE.md) - 原始函数对照
- [CHANGES_ANALYSIS.md](CHANGES_ANALYSIS.md) - 修改分析
- [HOP_NCFM_PIPELINE.md](HOP_NCFM_PIPELINE.md) - HoP-TM/NCFM流程
- [scripts/unified_eval.py](scripts/unified_eval.py) - 统一评估脚本

---

## 🎓 经验总结

### 成功经验
1. ✅ 统一数据接口设计正确，避免了8个不同的数据加载实现
2. ✅ 配置文件结构清晰，便于管理和校验
3. ✅ 运行器的阶段检查和资源检查有效防止了错误运行
4. ✅ Git忽略`raw/`的设计合理，保持了原始代码的完整性

### 教训
1. ⚠️ 数据Normalize应该在最后一步，避免重复变换
2. ⚠️ 跨算法的网络架构需要更严格的一致性验证
3. ⚠️ Seed控制应该从一开始就统一设计，而不是依赖各算法的默认行为
4. ⚠️ 路径管理应该集中化，避免HPC脚本和实际生成路径不一致

---

**结论**: 项目已完成"医疗数据适配层"和大部分运行框架，但需要解决NCFM pretrain和MTT/HoP-TM buffer的阻塞问题才能成为生产级Benchmark。
