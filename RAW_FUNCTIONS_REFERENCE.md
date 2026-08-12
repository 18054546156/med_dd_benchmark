# 原始算法函数对照表

本文档列出每个算法在原始`raw/`仓库中的核心函数及其作用。

---

## 1. DC/DSA/DM (raw/DatasetCondensation/)

### 主入口
- **main.py** - DC/DSA入口
- **main_DM.py** - DM入口

### utils.py 核心函数

#### 数据相关
- `get_dataset(dataset, data_path)` - 加载数据集（CIFAR10/100/SVHN等）
- `get_loops(ipc)` - 根据IPC确定outer_loop和inner_loop

#### 网络相关
- `get_default_convnet_setting()` - 获取ConvNet默认配置
- `get_network(model, channel, num_classes, im_size)` - 创建网络
- `get_eval_pool(eval_mode, model, model_eval)` - 获取评估网络池

#### 训练相关
- `epoch(mode, dataloader, net, optimizer, criterion, args, aug)` - 训练一个epoch
- `match_loss(gw_syn, gw_real, args)` - 计算梯度匹配损失
- `distance_wb(gwr, gws)` - 计算梯度距离

#### 评估相关
- `evaluate_synset(it_eval, net, images_train, labels_train, testloader, args)` - 评估合成数据集
  - 在合成数据上训练网络
  - 在测试集上评估accuracy

#### 数据增强
- `get_daparam(dataset, model, model_eval, ipc)` - 获取数据增强参数
- `DiffAugment(x, strategy, seed, param)` - 可微分数据增强
- `augment(images, dc_aug_param, device)` - 应用增强
- `rand_scale/rotate/flip/brightness/saturation/contrast/crop/cutout()` - 具体增强操作

#### 工具函数
- `get_time()` - 获取当前时间字符串
- `set_seed_DiffAug(param)` - 设置增强随机种子

---

## 2. DataDAM (raw/DataDAM/)

### 主入口
- **main_DataDAM.py** - DataDAM主入口

### utils.py 核心函数

#### DataDAM特有
- `get_attention(feature_set, param, exp, norm)` - **核心**: 生成空间注意力图
  - 从feature map提取attention
  - 用于匹配真实/合成数据的attention分布

#### 数据相关
- `get_dataset(dataset, data_path, args)` - 加载数据集（支持task_balance）

#### 网络相关
- `get_default_convnet_setting()` - ConvNet默认配置
- `get_network(model, channel, num_classes, im_size)` - 创建网络

#### 训练相关
- `epoch(mode, dataloader, net, optimizer, criterion, args, aug)` - 训练epoch
- `get_loops(ipc)` - 获取训练循环配置

#### 评估相关
- `evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, skip)` - 评估合成数据

#### 数据增强
- `get_daparam(dataset, model, model_eval, ipc)` - 数据增强参数
- `DiffAugment(x, strategy, seed, param)` - 可微分增强
- `augment(images, dc_aug_param, device)` - 应用增强
- `get_eval_pool(eval_mode, model, model_eval)` - 评估网络池
- `rand_*()` - 各种增强操作（同DC/DSA/DM）

#### 工具函数
- `get_time()` - 时间字符串
- `set_seed_DiffAug(param)` - 设置随机种子

---

## 3. CAFE (raw/CAFE/)

### 主入口
- **distill.py** - CAFE蒸馏入口
  - `build_logger(work_dir, cfgname)` - 构建日志器
  - `adjust_learning_rate(optimizer, epoch, init_lr)` - 调整学习率
  - `criterion_middle(real_feature, syn_feature)` - **核心**: 中间层特征匹配损失
  - `main()` - 主函数

### utils.py 核心函数

#### 数据相关
- `get_dataset(dataset, data_path)` - 加载数据集

#### 网络相关
- `get_default_convnet_setting()` - ConvNet默认配置
- `get_network(model, channel, num_classes, im_size)` - 创建网络

#### 训练相关
- `epoch(mode, dataloader, net, optimizer, criterion, args, aug)` - 训练epoch
- `match_loss(gw_syn, gw_real, args)` - 梯度匹配损失
- `distance_wb(gwr, gws)` - 梯度距离
- `get_loops(ipc)` - 训练循环配置

#### 评估相关
- `evaluate_synset(it_eval, net, images_train, labels_train, testloader, args)` - 评估合成数据
- `get_eval_pool(eval_mode, model, model_eval)` - 评估网络池

#### 数据增强
- `get_daparam(dataset, model, model_eval, ipc)` - 数据增强参数
- `DiffAugment(x, strategy, seed, param)` - 可微分增强
- `augment(images, dc_aug_param, device)` - 应用增强
- `rand_*()` - 各种增强操作

#### 工具函数
- `get_time()` - 时间字符串
- `set_seed_DiffAug(param)` - 设置随机种子

---

## 4. MTT (raw/mtt-distillation/)

### 主入口
- **buffer.py** - Expert buffer生成
  - `main(args)` - buffer生成主函数
- **distill.py** - 轨迹匹配蒸馏
  - `main(args)` - distill主函数

### utils.py 核心函数

#### 数据相关
- `get_dataset(dataset, data_path, batch_size, subset, args)` - 加载数据集
  - 支持subset参数（如imagenette）

#### 网络相关
- `get_default_convnet_setting()` - ConvNet默认配置
- `get_network(model, channel, num_classes, im_size, dist)` - 创建网络
  - `dist=True` 支持分布式训练

#### 训练相关
- `epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture)` - 训练epoch
  - 支持texture参数（纹理数据）

#### 评估相关
- `evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss, texture)` - 评估合成数据
  - 可选返回loss值

#### 数据增强
- `get_daparam(dataset, model, model_eval, ipc)` - 数据增强参数
- `DiffAugment(x, strategy, seed, param)` - 可微分增强
- `augment(images, dc_aug_param, device)` - 应用增强
- `get_eval_pool(eval_mode, model, model_eval)` - 评估网络池
- `rand_*()` - 各种增强操作

#### 工具函数
- `get_time()` - 时间字符串
- `set_seed_DiffAug(param)` - 设置随机种子

### 其他模块
- **reparam_module.py** - 参数重参数化模块（轨迹匹配核心）

---

## 5. HoP-TM (raw/HoP-TM/)

### 主入口
- **buffer/buffer_FTD.py** - FTD buffer生成（使用GSAM优化器）
- **distill/distill_high_order_spl.py** - 高阶轨迹蒸馏
  - `manual_seed(seed)` - 设置随机种子
  - `main(args)` - distill主函数

### utils/utils_gsam.py (Buffer阶段)

#### 数据相关
- `get_dataset(dataset, data_path, batch_size, args)` - 加载数据集

#### 网络相关
- `get_default_convnet_setting()` - ConvNet默认配置
- `get_network(model, channel, num_classes, im_size, dist)` - 创建网络

#### 训练相关
- `smooth_crossentropy(pred, gold, smoothing)` - 平滑交叉熵损失
- `epoch(mode, dataloader, net, optimizer, criterion, args, aug, scheduler, texture)` - 训练epoch
  - 支持scheduler参数（GSAM相关）

#### 评估相关
- `evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss, texture)` - 评估

#### 数据增强
- `get_daparam(dataset, model, model_eval, ipc)` - 数据增强参数
- `DiffAugment/augment/rand_*()` - 增强操作

#### 工具函数
- `get_time()` - 时间字符串
- `get_eval_pool(eval_mode, model, model_eval)` - 评估网络池

### utils/utils_baseline.py (Distill阶段)

#### 额外函数
- `reduce_dataset(train_set, rate, class_num, num_per_class)` - 数据集降采样

#### 网络相关
- `get_network(model, channel, num_classes, im_size, dist)` - 创建网络

#### 训练相关
- `epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture, If_Float)` - 训练epoch

#### 评估相关
- `evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss, texture, train_criterion, Preciser_Scheduler, type)` - 评估合成数据
  - 支持更精细的scheduler和type参数
- `evaluate_baseline(it_eval, net, trainloader, testloader, args, ...)` - 评估baseline（真实数据）

#### 数据增强
- `get_daparam/DiffAugment/augment/rand_*()` - 同utils_gsam

### GSAM模块 (buffer/gsam/)
- **gsam.py** - Generalized SAM优化器
- **scheduler.py** - 学习率调度器
- **util.py** - GSAM工具函数
- **wide_res_net.py** - WideResNet实现

### 工具模块 (buffer/utility/)
- **initialize.py** - 网络初始化
- **cutout.py** - Cutout数据增强
- **bypass_bn.py** - Batch Normalization绕过

---

## 6. NCFM (raw/NCFM/)

### 主入口
- **pretrain/pretrain_script.py** - 预训练脚本
- **condense/condense_script.py** - 蒸馏脚本
- **evaluation/evaluation_script.py** - 评估脚本

### utils/utils.py 核心函数

#### 网络相关
- `define_model(dataset, norm_type, net_type, nch, depth, width, nclass, logger, size)` - **核心**: 定义模型
  - 支持多种norm_type（batch/instance/group等）
  - 支持多种net_type（convnet/resnet等）
  - 灵活配置depth和width

#### 数据相关
- `load_resized_data(...)` - 加载并调整数据集大小
- `get_loader(args)` - 获取数据加载器

#### 优化器相关
- `get_optimizer(optimizer, parameters, lr, mom_img, weight_decay, logger)` - 获取优化器
  - 支持SGD/Adam等

#### 特征提取相关（NCFM核心）
- `get_feature_extractor(args)` - 获取特征提取器
  - 从预训练模型提取特征
- `update_feature_extractor(args, model_init, model_final, model_interval, a, b)` - 更新特征提取器
  - model_init: 初始模型
  - model_final: 最终训练后模型
  - model_interval: 中间checkpoint模型
  - 用于计算Neural Characteristic Function

#### 可视化相关
- `get_plotter(args)` - 获取可视化plotter

#### 工具函数
- `apply_blurpool(mod)` - 应用blur pooling
- `BlurPoolConv2d` (class) - Blur pooling卷积层

### condenser/Condenser.py
- **Condenser** (class) - NCFM核心蒸馏类
  - `load_condensed_data()` - 加载合成数据
  - `condense()` - 执行蒸馏
  - 内部实现Neural Collapse特征匹配

### condenser/compute_loss.py
- `neural_collapse_loss()` - Neural Collapse损失函数

### condenser/evaluate.py
- `evaluate_synset()` - 评估合成数据集

### 其他模块
- **utils/init_script.py** - 初始化脚本（DDP配置等）
- **utils/diffaug.py** - 可微分数据增强
- **utils/ddp.py** - 分布式数据并行工具
- **utils/mix_cut_up.py** - MixUp/CutMix等混合增强
- **utils/experiment_tracker.py** - 实验追踪

---

## 函数分类总结

### 通用函数（所有算法都有）

#### 数据加载
```python
get_dataset(dataset, data_path, ...)
```
- **作用**: 加载训练/测试数据
- **输出**: channel, im_size, num_classes, dst_train, dst_test, testloader

#### 网络创建
```python
get_network(model, channel, num_classes, im_size, ...)
```
- **作用**: 创建ConvNet/ResNet等网络
- **输出**: 网络实例

#### 训练循环
```python
epoch(mode, dataloader, net, optimizer, criterion, args, aug, ...)
```
- **作用**: 训练或测试一个epoch
- **输出**: loss, accuracy

#### 评估合成数据
```python
evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, ...)
```
- **作用**: 在合成数据上训练网络，在测试集上评估
- **流程**:
  1. 创建网络
  2. 在合成数据上训练 epoch_eval_train 轮
  3. 在测试集上测试
  4. 返回accuracy
- **输出**: accuracy (0-100)

#### 数据增强
```python
DiffAugment(x, strategy, seed, param)
get_daparam(dataset, model, model_eval, ipc)
augment(images, dc_aug_param, device)
```
- **作用**: 可微分数据增强

#### 评估网络池
```python
get_eval_pool(eval_mode, model, model_eval)
```
- **作用**: 获取评估用的网络架构列表
- **eval_mode**:
  - 'S': 单一架构（同训练网络）
  - 'M': 多架构
  - 'W': 不同width
  - 'D': 不同depth

### 算法特有函数

#### DC/DSA/DM
```python
match_loss(gw_syn, gw_real, args)  # 梯度匹配损失
distance_wb(gwr, gws)              # 梯度距离
get_loops(ipc)                     # outer/inner loop配置
```

#### DataDAM
```python
get_attention(feature_set, param, exp, norm)  # 空间注意力图生成
```

#### CAFE
```python
criterion_middle(real_feature, syn_feature)   # 中间层特征匹配损失
```

#### MTT/HoP-TM
```python
# 使用reparam_module.py进行参数重参数化
# 轨迹匹配在distill.py中实现
```

#### HoP-TM特有
```python
smooth_crossentropy(pred, gold, smoothing)    # 平滑交叉熵
evaluate_baseline(...)                        # 评估真实数据baseline
reduce_dataset(...)                           # 数据集降采样
```

#### NCFM特有
```python
define_model(...)                             # 灵活的模型定义
get_feature_extractor(args)                   # 获取特征提取器
update_feature_extractor(...)                 # 更新特征提取器（Neural Collapse核心）
load_resized_data(...)                        # 加载调整大小后的数据
```

---

## evaluate_synset() 详细对比

这是**评估公平性的关键函数**，各算法实现略有不同：

### DC/DSA/DM (raw/DatasetCondensation/utils.py:337)
```python
def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    # 参数:
    # - it_eval: 当前迭代数
    # - net: 评估网络（单个）
    # - images_train: 合成图像
    # - labels_train: 合成标签
    # - testloader: 测试集加载器
    # - args.epoch_eval_train: 训练轮数（默认300）
    
    # 流程:
    net_eval = get_network(net, channel, num_classes, im_size)
    for ep in range(args.epoch_eval_train):
        train_on_synthetic_data(net_eval, images_train, labels_train)
    acc = test(net_eval, testloader)
    return acc
```

### DataDAM (raw/DataDAM/utils.py:423)
```python
def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, skip=False):
    # 额外参数:
    # - skip: 是否跳过评估
    
    # 流程: 同DC/DSA/DM
    # args.epoch_eval_train: 默认1800（更长！）
```

### MTT (raw/mtt-distillation/utils.py:352)
```python
def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, texture=False):
    # 额外参数:
    # - return_loss: 是否返回loss值
    # - texture: 是否使用纹理数据
    
    # 流程: 同基础版本
    # 可选返回 (acc, loss)
```

### CAFE (raw/CAFE/utils.py:305)
```python
def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    # 同DC/DSA/DM
    # args.epoch_eval_train: 默认300
```

### HoP-TM (raw/HoP-TM/utils/utils_baseline.py:377)
```python
def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, 
                    return_loss=False, texture=False, train_criterion=None, 
                    Preciser_Scheduler=False, type=1):
    # 额外参数:
    # - train_criterion: 自定义训练损失函数
    # - Preciser_Scheduler: 是否使用更精细的学习率调度
    # - type: 评估类型（1或其他）
    
    # 流程: 更复杂的训练策略
```

### NCFM (raw/NCFM/condenser/evaluate.py)
```python
def evaluate_synset(...):
    # 独立的evaluation stage
    # 使用不同的评估协议
    # 支持val_repeat参数（重复多次取平均）
```

---

## 关键差异点

### 1. evaluate_synset的epoch_eval_train参数
```
DC/DSA/DM:  默认 300
DataDAM:    默认 1800  ← 训练更久
CAFE:       默认 300
MTT:        默认 1000
HoP-TM:     默认 ?
NCFM:       独立stage，参数不同
```

### 2. evaluate_synset的调用时机
- **DC/DSA/DM/DataDAM/CAFE/MTT/HoP-TM**: 在蒸馏过程中周期性调用
- **NCFM**: 独立的evaluation stage

### 3. 评估网络数量 (num_eval)
```python
for _ in range(args.num_eval):
    acc = evaluate_synset(...)
    accs.append(acc)
final_acc = mean(accs)
```
- **不同算法的num_eval不同** → 结果统计方式不同

---

## 统一评估的关键

要实现公平评估，需要：

1. **统一的evaluate_synset()实现**
   - 固定 epoch_eval_train（如1000）
   - 固定 optimizer配置（SGD, lr=0.01, momentum=0.9）
   - 固定 batch_size
   - 固定 数据增强策略

2. **统一的num_eval**
   - 固定评估网络数量（如5）
   - 固定随机种子策略

3. **统一的测试集**
   - 所有算法使用相同的test split

4. **独立的评估脚本**
   - 不依赖各算法内嵌的evaluation
   - 读取任意算法生成的DD数据
   - 输出标准化的accuracy报告
