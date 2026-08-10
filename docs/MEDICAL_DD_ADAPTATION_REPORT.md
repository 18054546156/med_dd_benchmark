# 医疗数据集 DD 适配与验证报告

更新时间：2026-08-10

## 结论

本仓库已经完成 8 个数据集蒸馏/数据集压缩算法对 3 个医疗数据集的适配，并完成官方最小入口 smoke：

| 验证层级 | 结果 | 说明 |
|---|---:|---|
| 数据集 loader 合同 | 24/24 PASS | 8 个算法 × 3 个数据集 |
| 算法特定 one-step probe | 24/24 PASS | DC/DSA/DM/DataDAM/CAFE 调用各自匹配损失；MTT/HoP-TM/NCFM 完成对应网络/轨迹探针 |
| 官方最小入口 | 24/24 PASS | 使用各算法自己的 buffer/config/主程序，保存真实入口产物 |
| NCFM 专项 loader 测试 | 3/3 PASS | PathMNIST、COVID、Kvasir |
| Python 编译检查 | PASS | `compileall` 无错误 |

这里的“官方最小入口”是可复现性和代码正确性的 smoke，不是论文最终结果：使用 IPC=1、Iteration=0 或 1 次更新、1 个 expert/epoch 等小参数。完整论文实验仍应使用各算法的 `*_full.yaml` 和原论文迭代规模，并单独记录准确率。

## 0. 文件级修改清单

下面的清单以外层仓库原有 checkpoint `5574aa5` 为基线，覆盖已提交的适配内容以及当前工作树中针对 smoke 和 Windows 运行补充的修复。`raw/` 下的六个官方仓库没有改动。

### 共享数据层、验证和文档

| 文件 | 修改内容 |
|---|---|
| `utils/medical_dataset_utils.py` | 增加三套数据集规格、PathMNIST 缓存路径解析、`scalarize_label`、`MedMNISTWrapper`、合同验证和中文注释；MedMNIST 的 `array([k])` 统一为 Python `int`。 |
| `scripts/prepare_medical_data.py` | 实现官方数据下载、解压、准备目录、固定划分、类别检查、MD5/SHA256 和 `manifest.json`。 |
| `scripts/validate_medical_adapters.py` | 在隔离子进程中检查 8 个算法 × 3 个数据集的 loader、图像尺寸、类别数、单样本标签和 batch 标签 dtype/shape。 |
| `scripts/run_medical_one_step.py` | 让 24 个组合各执行一次真实网络前向和反向更新；DC/DSA/DM/DataDAM/CAFE 使用各自核心匹配项，MTT/HoP-TM/NCFM 作为快速网络/轨迹探针。 |
| `scripts/validate_official_smoke.py` | 只读审计 24 个官方入口组合的日志阶段标记和算法特有 buffer/result/pretrain/condensed 产物。 |
| `test_ncfm_adapted.py` | 增加 NCFM 三个医疗数据集的独立 loader/标量标签检查。 |
| `configs/config_loader.py` | 统一 6 个算法的 dataset 子目录配置路径，修正 DC/DSA/DM 的 `ipc*_dc_*` 文件名和 NCFM 的路径解析，保证 18 套正式配置可被统一 API 找到。 |
| `docs/DATASET_CONTRACT.md` | 固化官方下载来源、原始格式、prepared 目录、标签合同、尺寸、类别、归一化和各算法入口边界。 |
| `.gitignore` | 忽略下载缓存、prepared 大文件、buffer、结果、W&B 和运行日志，避免把实验产物混入源码提交。 |

### DC / DSA / DM

| 文件 | 修改内容 |
|---|---|
| `adapted/dc_dsa_dm/utils.py` | 增加 MedMNIST/ImageFolder 三数据集 loader，动态返回通道、尺寸、类别数和类别 loader；PathMNIST 标签标量化；修复 Siamese augmentation 的重叠内存赋值。 |
| `adapted/dc_dsa_dm/main.py` | 保留 DC/DSA 原始 gradient matching 流程；接入动态网络规格，增加 `--fast_eval` 和 `--device auto|cuda|cpu`。 |
| `adapted/dc_dsa_dm/main_DM.py` | 保留 DM 的 embedding/distribution matching；初始化 `dc_aug_param`，接入动态设备和医疗规格。 |
| `configs/dc_dsa_dm/{pathmnist,covid,kvasir}/` | 提供三数据集的 DC/DSA/DM 正式配置和快速配置。 |

### MTT

| 文件 | 修改内容 |
|---|---|
| `adapted/mtt/utils.py` | 统一三数据集 loader 返回合同，生成标量标签、动态类别 loader 和动态 `channel/im_size/num_classes`。 |
| `adapted/mtt/distill.py` | 保持原始 expert trajectory matching；接入医疗尺寸/类别、augmentation 参数和离线结果目录；保存阶段使用当前 device，支持 CPU smoke。buffer 入口直接复用同一 `get_dataset`。 |
| `configs/mtt/{pathmnist,covid,kvasir}/` | 提供 MTT 的三数据集配置。 |

### HoP-TM

| 文件 | 修改内容 |
|---|---|
| `adapted/hop_tm/utils/utils_baseline.py` | baseline loader 支持三数据集、标量标签、identity `class_map` 和动态类别/尺寸。 |
| `adapted/hop_tm/utils/utils_gsam.py` | GSAM/FTD loader 与 baseline 保持相同医疗合同。 |
| `adapted/hop_tm/buffer/buffer_FTD.py` | 让 FTD buffer 使用医疗 loader 和独立的 `*_NO_ZCA` buffer 路径。 |
| `adapted/hop_tm/distill/distill_high_order_spl.py` | 保留高阶轨迹/角度损失；支持两阶段配置解析、CLI 覆盖 `Iteration`/`syn_steps`、医疗设备和当前 device 保存。 |
| `adapted/hop_tm/distill/{DATM,DATM_cal_time,DATM_tesla}.py` | 同步医疗 buffer 路径和数据合同。 |
| `adapted/hop_tm/utils/cfg.py` | 修复 `None`、bool 和可覆盖整数配置解析，并允许医疗配置字段。 |
| `configs/hop_tm/{pathmnist,covid,kvasir}/` | 提供 `ipc10_full.yaml` 和 `ipc1_smoke.yaml`；修复 Kvasir YAML 断行。 |

### NCFM

| 文件 | 修改内容 |
|---|---|
| `adapted/ncfm/utils/utils.py` | 增加三数据集路径解析、transform、类别元数据、标量 `targets`、PathMNIST wrapper 和 CPU 可运行的 loader。 |
| `adapted/ncfm/data/dataloader.py` | 让 `ClassDataLoader`/`ClassMemDataLoader` 使用传入 device，不再无条件访问 CUDA。 |
| `adapted/ncfm/utils/ddp.py` | 在 CPU/Windows smoke 下允许缺省 local-rank 环境并返回 CPU device；CUDA 环境仍设置对应 GPU。 |
| `adapted/ncfm/condense/condense_script.py` | 接入医疗配置和 condensation 路径，保留 NCFM 原始 match/calibration loss，并把 Condenser 设备改为跟随 `args.device`。 |
| `adapted/ncfm/condenser/condense_transfom.py` | 增加 PathMNIST、COVID、Kvasir 的尺寸/变换，并避免已经归一化的医疗合成张量重复归一化。 |
| `adapted/ncfm/utils/init_script.py` | Windows 无 NCCL 时 fallback 到 Gloo，Linux 可用 NCCL 时仍保留官方行为。 |
| `adapted/ncfm/utils/experiment_tracker.py` | 修复离线运行时结果目录和 run name 为空的问题。 |
| `configs/ncfm/{pathmnist,covid,kvasir}/` | 提供原始规模 `ipc10_full.yaml` 和最小入口 `ipc1_smoke.yaml`。 |

### DataDAM 和 CAFE

| 算法 | 文件 | 修改内容 |
|---|---|---|
| DataDAM | `adapted/datadam/utils.py` | 三数据集 loader、PathMNIST 标签标量化、动态类别/尺寸。 |
| DataDAM | `adapted/datadam/main_DataDAM.py` | 保留 attention/output 双重 matching；修复动态目录创建、CPU/GPU loss 设备和 `.cpu()` 截断 `image_syn` 梯度的问题。 |
| DataDAM | `configs/datadam/{pathmnist,covid,kvasir}/` | 三数据集正式配置。 |
| CAFE | `adapted/cafe/utils.py` | 三数据集 loader、PathMNIST 标签标量化、动态类别/尺寸和 Siamese augmentation 兼容修复。 |
| CAFE | `adapted/cafe/distill.py` | 保留多层 feature alignment 和 inner-loop；按运行时类别数 reshape，使用动态 `[N,C,H,W]`，增加 `--smoke`。 |
| CAFE | `configs/cafe/{pathmnist,covid,kvasir}/` | 三数据集正式配置。 |

当前工作树中额外的入口修复集中在 `adapted/cafe/distill.py`、`adapted/dc_dsa_dm/main.py`、`adapted/dc_dsa_dm/main_DM.py`、`adapted/mtt/distill.py`、`adapted/hop_tm/distill/distill_high_order_spl.py`、`adapted/hop_tm/utils/cfg.py`、`adapted/ncfm/condense/condense_script.py`、`adapted/ncfm/condenser/condense_transfom.py`、`adapted/ncfm/utils/{ddp,experiment_tracker,init_script}.py`、`configs/config_loader.py` 和 `scripts/run_medical_one_step.py`；这些是对已完成适配的可运行性补强，不是替换任何官方算法损失。

### 函数级输入/输出合同

| 算法 | 关键函数 | 输入 -> 输出 |
|---|---|---|
| DC/DSA | `dc_dsa_dm/utils.py:get_dataset`；`match_loss` | `(dataset, data_path)` -> 9 项数据合同；`(gw_syn, gw_real, args)` -> 标量梯度匹配损失 |
| DM | `dc_dsa_dm/main_DM.py`；`utils.py:get_network` | `get_dataset` 的真实 batch 和 `image_syn` -> embedding 均值匹配标量；网络函数 `(model, channel, classes, im_size)` -> `nn.Module` |
| MTT | `mtt/utils.py:get_dataset`；`buffer.py`/`distill.py` | `(dataset, data_path, batch_size, args)` -> 12 项合同；真实训练 -> `[expert][epoch][parameter tensors]` buffer；buffer + synthetic batch -> trajectory loss |
| HoP-TM | `hop_tm/utils/{utils_baseline,utils_gsam}.py:get_dataset`；`distill_high_order_spl.py` | `(dataset, data_path, batch_size, args)` -> 11 项合同；FTD buffer + synthetic batch -> parameter loss + high-order angle loss |
| NCFM | `ncfm/utils/utils.py:load_resized_data/get_loader`；`Condenser.condense` | `(dataset, data_dir, size, nclass)` -> `(train_dataset, val_dataset)`；`args.run_mode` -> 对应 loader；预训练模型和 class loader -> `data_init.pt` |
| DataDAM | `datadam/utils.py:get_dataset/get_attention` | `(dataset, data_path, args)` -> 10 项合同；feature map -> attention map；real/synthetic attention + output -> image update |
| CAFE | `cafe/utils.py:get_dataset`；`distill.py:criterion_middle` | `(dataset, data_path)` -> 9 项合同；同类 feature map -> 类中心 MSE；多层 feature + inner output -> image update |

所有 loader 合同中的单样本标签都是 Python `int`，batch 标签都是一维 `torch.long`；完整字段顺序和每个算法的额外返回项见各节说明。

## 1. 数据合同与官方下载

统一合同位于 [`docs/DATASET_CONTRACT.md`](DATASET_CONTRACT.md)，固定属性位于 [`utils/medical_dataset_utils.py`](../utils/medical_dataset_utils.py) 的 `MEDICAL_DATASET_SPECS`。

| 数据集 | 官方来源 | 原始格式 | prepared 输入 | 类别 | train/test |
|---|---|---|---|---:|---:|
| PathMNIST | [MedMNIST](https://medmnist.com/) | `pathmnist.npz` | `3x32x32` | 9 | 89996 / 7180 |
| COVID | [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | 类别目录 + `images/` | `3x112x112` | 4 | 16933 / 4232 |
| Kvasir | [Simula Kvasir v2](https://datasets.simula.no/kvasir/) | 8 类 ImageFolder | `3x128x128` | 8 | 6400 / 1600 |

目录约定：

```text
data/raw/                         # 官方下载包和解压目录，不进入源码提交
data/prepared/PathMNIST/          # pathmnist.npz + manifest.json
data/prepared/COVID/train/<class>/
data/prepared/COVID/test/<class>/
data/prepared/Kvasir/train/<class>/
data/prepared/Kvasir/test/<class>/
```

准备脚本是 [`scripts/prepare_medical_data.py`](../scripts/prepare_medical_data.py)。COVID 只使用 `images/`，同目录的 masks 不作为分类样本；COVID/Kvasir 使用 ImageFolder 目录名产生类别索引。

归一化约定：PathMNIST 使用合同中的 `[0.741, 0.533, 0.706]` / `[0.402, 0.821, 0.407]`；COVID 和 Kvasir 使用 ImageNet mean/std。后两者是 benchmark 工程约定，不声称是数据集官方统计量。

## 2. 统一数据层修改

### `utils/medical_dataset_utils.py`

- 定义三套 `MEDICAL_DATASET_SPECS`。
- `get_medmnist_root(data_path)` 统一 PathMNIST 缓存位置。
- `scalarize_label(label)` 将 `ndarray([k])`、单元素 Tensor 和 Python 数统一为 `int`。
- `MedMNISTWrapper.__getitem__` 保证单样本标签是标量整数；DataLoader 后保证标签形状为 `[B]`、类型为 `torch.long`。
- 提供数据合同验证、类别名和 manifest 辅助函数。

### `scripts/prepare_medical_data.py`

- 封装 PathMNIST、COVID、Kvasir 的官方下载/解压/复制/划分流程。
- 生成可审计的 `manifest.json`，记录来源、随机种子、划分比例和类别计数。

## 3. 各算法流程、输入输出和修改

### 3.1 DC / DSA / DM

入口：[`adapted/dc_dsa_dm/main.py`](../adapted/dc_dsa_dm/main.py)；DM 入口为 [`main_DM.py`](../adapted/dc_dsa_dm/main_DM.py)。

共同数据函数：

```text
get_dataset(dataset, data_path)
  -> channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader
```

主流程：

```text
get_dataset
  -> images_all [N,C,H,W], labels_all [N]
  -> image_syn [num_classes*ipc,C,H,W], label_syn [num_classes*ipc]
  -> DC: gradient matching
     DSA: gradient matching + differentiable Siamese augmentation
     DM: distribution/feature mean matching
  -> evaluate_synset
  -> res_<method>_<dataset>_<model>_<ipc>ipc.pt + vis_*.png
```

修改内容：

- `utils.py` 增加 PathMNIST MedMNISTWrapper、COVID/Kvasir ImageFolder 分支和医疗尺寸/类别/归一化属性。
- 移除对固定 CIFAR-10 类别数和尺寸的依赖。
- 所有 active DSA augmentation 的原地赋值改为 `.clone()`，兼容 PyTorch 2.5。
- `main.py` 增加 `--fast_eval`。默认仍使用原始 DSA 评估轮数；smoke 可显式使用命令行指定的少量评估 epoch。
- `main_DM.py` 初始化 `args.dc_aug_param`，修复 DM 第一次评估访问不存在属性的问题。
- 中文注释说明医疗数据分支、标签标量化、动态尺寸和 DM augmentation 合同。

核心配置：`dataset`、`model`、`ipc`、`Iteration`、`lr_img`、`lr_net`、`batch_real`、`batch_train`、`init`、`dsa_strategy`、`data_path`、`save_path`、`eval_mode`、`num_eval`、`epoch_eval_train`。

### 3.2 MTT

入口：[`adapted/mtt/buffer.py`](../adapted/mtt/buffer.py) 和 [`adapted/mtt/distill.py`](../adapted/mtt/distill.py)。

```text
buffer.py
  -> get_dataset(...)
  -> teacher network 在真实数据上训练
  -> replay_buffer_0.pt: [expert][epoch][parameter tensors]

distill.py
  -> 读取 replay buffer
  -> image_syn/label_syn
  -> synthetic gradient update 对齐 expert trajectory
  -> evaluate_synset
  -> logged_files/<dataset>/{images,labels}*.pt
```

`mtt/utils.py` 的 `get_dataset` 返回 12 项：`channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv`。

修改内容：

- PathMNIST 标签使用 `MedMNISTWrapper` 标量化；COVID/Kvasir 使用 ImageFolder。
- buffer 和 distill 都使用动态 `channel/im_size/num_classes`。
- 补齐 `dc_aug_param` 初始化和 `get_daparam` 导入。
- 离线 W&B 没有 run name 时生成稳定的本地结果目录。
- 保留原始 trajectory matching，不将 MTT 改写成普通梯度匹配。

关键配置：`num_experts`、`train_epochs`、`buffer_path`、`syn_steps`、`expert_epochs`、`Iteration`、`ipc`、`lr_img`、`lr_lr`、`batch_real`、`batch_train`、`load_all`。

证据：

```text
buffers/official_mtt/PathMNIST/ConvNet/replay_buffer_0.pt
buffers/official_mtt/COVID/ConvNet/replay_buffer_0.pt
buffers/official_mtt/Kvasir/ConvNet/replay_buffer_0.pt
results/official_smoke/current_mtt/PathMNIST.stdout.log
results/official_smoke/current_mtt/COVID.stdout.log
results/official_smoke/current_mtt/Kvasir.stdout.log
```

三套当前 smoke 都加载对应 `buffers/official_mtt/<dataset>/ConvNet/replay_buffer_0.pt`，进入 `training begins`、`iter = 0000`，并完成 trajectory matching 更新与 synthetic 保存；本轮 `num_eval=0`，不把它当作准确率实验。

### 3.3 HoP-TM

入口：[`buffer/buffer_FTD.py`](../adapted/hop_tm/buffer/buffer_FTD.py) 和 [`distill/distill_high_order_spl.py`](../adapted/hop_tm/distill/distill_high_order_spl.py)。

```text
buffer_FTD.py
  -> utils_gsam.get_dataset
  -> GSAM/FTD teacher training
  -> replay_buffer_0.pt

distill_high_order_spl.py
  -> utils_baseline.get_dataset
  -> real images + scalar labels
  -> read expert trajectory
  -> syn_steps inner updates
  -> parameter loss + high-order angle loss
  -> logged_files/<dataset>/...
```

`utils_baseline.py`、`utils_gsam.py` 和相关 evaluation 入口返回 11 项：`channel, im_size, num_classes, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv`。

修改内容：

- 两套 loader 都支持 PathMNIST、COVID、Kvasir，并使用 identity `class_map`。
- FTD/GSAM buffer 的医疗分支固定 `*_NO_ZCA` 路径，避免把未做 ZCA 的医疗数据误读为 CIFAR buffer。
- `distill_high_order_spl.py` 改为两阶段 parser：先读取 cfg，再允许 CLI 覆盖 `Iteration`、`syn_steps` 等参数。
- `cfg.py` 允许医疗配置扩展字段；修复 `None`、bool 和可覆盖整数参数的解析。
- 重写 Kvasir YAML 的断行错误，保证 `expert_epochs`、`lr_img`、`Iteration` 等是独立字段。
- 增加中文注释，解释 `class_map`、医疗 buffer 路径和配置解析行为。

关键配置：`high_order`、`base_threshold`、`growing_factor`、`lamb`、`syn_steps`、`expert_epochs`、`min_start_epoch`、`max_start_epoch`、`buffer_path`、`data_path`、`ipc`、`Iteration`。

证据：

```text
buffers/official_smoke/hop_full/PathMNIST_NO_ZCA/ConvNet/replay_buffer_0.pt
buffers/official_smoke/hop_full/COVID_NO_ZCA/ConvNet/replay_buffer_0.pt
buffers/official_smoke/hop/Kvasir_NO_ZCA/ConvNet/replay_buffer_0.pt
results/official_smoke/current_hop/PathMNIST.stdout.log
results/official_smoke/current_hop/COVID.stdout.log
results/official_smoke/current_hop/Kvasir.stdout.log
```

三套当前日志都包含 `training begins`、绝对 expert 定位和 `iter = 0000` 的 high-order 更新，命令退出码为 0；本轮为 `skip_first_eva=True` 的更新 smoke，不用它宣称最终准确率。

### 3.4 NCFM

入口：[`pretrain/pretrain_script.py`](../adapted/ncfm/pretrain/pretrain_script.py) 和 [`condense/condense_script.py`](../adapted/ncfm/condense/condense_script.py)。

```text
pretrain_script.py
  -> load_resized_data / get_loader
  -> define_model
  -> premodel0_init.pth.tar
  -> premodel0_trained.pth.tar

condense_script.py
  -> ClassDataLoader / ClassMemDataLoader
  -> Condenser.load_condensed_data
  -> NCFM match loss + calibration loss
  -> Condenser.condense
  -> results/.../distilled_data/data_init.pt
```

`load_resized_data` 返回 `(train_dataset, val_dataset)`；`get_loader` 按 `run_mode` 返回 condense loader、validation loader 或 `(train_loader, val_loader, train_sampler)`。

修改内容：

- 增加 PathMNIST/COVID/Kvasir 数据入口、路径解析、normalize、类别元数据和标量标签。
- `ClassDataLoader`/`ClassMemDataLoader` 使用可传入的 device，避免 CPU smoke 无条件访问 CUDA。
- Windows 无 NCCL 时在 `init_script.py` 自动 fallback 到 Gloo；Linux 且 NCCL 可用时保留官方 NCCL 行为。
- `accuracy` 将 top-k 截断到 logits 类别数，修复 COVID 四分类请求 top-5 的越界异常。
- 添加 `configs/ncfm/{pathmnist,covid,kvasir}/ipc1_smoke.yaml`，只用于最小入口证据；原有 `ipc10_full.yaml` 保留完整实验规模。
- 中文注释解释 dataset root、targets、分布式后端和 top-k 行为。

关键配置：`backend`、`workers`、`dataset`、`nclass`、`size`、`data_dir`、`nch`、`net_type`、`depth`、`width`、`pertrain_epochs`、`model_num`、`batch_size`、`pretrain_dir`、`num_premodel`、`niter`、`iter_calib`、`num_freqs`、`factor`、`ipc`。

证据：

```text
results/official_smoke/current_ncfm/{pathmnist,covid,kvasir}-pretrain-final.log
results/official_smoke/current_ncfm/{pathmnist,covid,kvasir}-condense-final.log
results/official_smoke/ncfm_pretrained/<dataset>/premodel0_{init,trained}.pth.tar
results/official_smoke/ncfm/<dataset>/condense/.../distilled_data/data_init.pt
```

三套当前 direct pretrain 和三套 direct condense 均返回 `EXIT_CODE=0`；单进程入口自动使用 Gloo + `TCPStore(use_libuv=False)`，多进程 Linux 仍可使用原始 `env://`/NCCL。

### 3.5 DataDAM

入口：[`adapted/datadam/main_DataDAM.py`](../adapted/datadam/main_DataDAM.py)。

```text
get_dataset(dataset, data_path, args)
  -> channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader, zca
  -> images_all/labels_all
  -> output classification matching + attention matching
  -> evaluate_synset
  -> res_DataDAM_*.pt
```

修改内容：

- 添加三套医疗 loader、动态输入尺寸和动态类别数。
- PathMNIST 标签通过 wrapper 标量化。
- attention loss 保持 `image_syn` 在当前设备，移除会截断梯度的 `.cpu()`。
- 删除/绕过固定 CIFAR-10 reshape 和类别假设。
- 保留 DataDAM 的 attention/output 双重匹配流程，不改成 DC。

关键配置：`ipc`、`Iteration`、`lr_img`、`lr_net`、`batch_real`、`batch_train`、`init`、`task_balance`、`zca`、`num_exp`、`num_eval`。

证据：

```text
results/smoke/official_DataDAM_PathMNIST_eval/res_DataDAM_PathMNIST_ConvNet_1ipc_.pt
results/smoke/official_DataDAM_COVID/res_DataDAM_COVID_ConvNet_1ipc_.pt
results/smoke/official_DataDAM_Kvasir/res_DataDAM_Kvasir_ConvNet_1ipc_.pt
results/smoke/official_DataDAM_{PathMNIST,COVID,Kvasir}_*.stdout.log
```

三套日志均进入 `training begins`、`iter = 0000` 和 final results；PathMNIST/COVID/Kvasir 的 1-epoch eval test acc 分别为 0.1568、0.0990、0.1344。

### 3.6 CAFE

入口：[`adapted/cafe/distill.py`](../adapted/cafe/distill.py)。

```text
get_dataset(dataset, data_path)
  -> channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader
  -> feature hooks / middle feature alignment
  -> output classification + feature matching
  -> inner-loop synthetic update
  -> res_*.pt + vis_*.png
```

修改内容：

- 添加三套医疗 loader、标签标量化和动态类别数/尺寸。
- `criterion_middle` 按运行时 `num_classes` reshape，不固定 10 类。
- 所有合成图像和 real batch 使用 `[N,C,H,W]` 动态拼接，不固定 `3x32x32`。
- 保留 CAFE 的多层 feature alignment + inner-loop 流程。

注意：CAFE 继承了原始脚本的 `args.method='DC'` 结果文件前缀，因此文件名可能是 `res_DC_<dataset>...pt`；内容和日志仍来自 `adapted/cafe/distill.py` 的 CAFE feature matching，不代表调用了 DC 主程序。

关键配置：`ipc`、`Iteration`、`lr_img`、`lr_net`、`batch_real`、`batch_train`、`init`、`fourth_weight`、`third_weight`、`second_weight`、`first_weight`、`inner_weight`、`lambda_1`、`lambda_2`。

证据：

```text
results/smoke/official_CAFE_PathMNIST/res_DC_PathMNIST_ConvNet_1ipc.pt
results/smoke/official_CAFE_COVID/res_DC_COVID_ConvNet_1ipc.pt
results/smoke/official_CAFE_Kvasir/res_DC_Kvasir_ConvNet_1ipc.pt
results/smoke/official_CAFE_{PathMNIST,COVID,Kvasir}_v1.*.log
```

三套 CAFE 日志均有 1-epoch eval、`iter = 0000` 和 final results；test acc 分别为 0.1369、0.4823、0.1394。

## 4. 官方最小入口矩阵

| 算法 | PathMNIST | COVID | Kvasir | 证据类型 |
|---|---|---|---|---|
| DC | PASS | PASS | PASS | `main.py`，结果文件 |
| DSA | PASS | PASS | PASS | `main.py --method DSA`，结果文件 |
| DM | PASS | PASS | PASS | `main_DM.py`，结果文件；COVID/Kvasir 为 train-only smoke |
| MTT | PASS | PASS | PASS | buffer + trajectory distill |
| HoP-TM | PASS | PASS | PASS | FTD buffer + high-order distill |
| NCFM | PASS | PASS | PASS | pretrain + condense |
| DataDAM | PASS | PASS | PASS | attention matching + eval |
| CAFE | PASS | PASS | PASS | feature matching + eval |

### 4.1 产物级证据路径

以下文件是已经实际生成的最小入口产物，不是只检查 import 的占位文件：

| 算法 | PathMNIST | COVID | Kvasir |
|---|---|---|---|
| DC | `results/official_smoke/dc_pathmnist/res_DC_PathMNIST_ConvNet_1ipc.pt` | `results/official_smoke/dc_covid_v2/res_DC_COVID_ConvNet_1ipc.pt` | `results/official_smoke/dc_kvasir/res_DC_Kvasir_ConvNet_1ipc.pt` |
| DSA | `results/official_smoke/dsa_pathmnist/res_DSA_PathMNIST_ConvNet_1ipc.pt` | `results/official_smoke/dsa_covid_v2/res_DSA_COVID_ConvNet_1ipc.pt` | `results/official_smoke/dsa_kvasir/res_DSA_Kvasir_ConvNet_1ipc.pt` |
| DM | `results/official_smoke/dm_pathmnist_v2/res_DM_PathMNIST_ConvNet_1ipc.pt` | `results/official_smoke/dm_covid_v3/res_DM_COVID_ConvNet_1ipc.pt` | `results/official_smoke/dm_kvasir/res_DM_Kvasir_ConvNet_1ipc.pt` |
| MTT | `buffers/official_mtt/PathMNIST/ConvNet/replay_buffer_0.pt` + `results/official_smoke/current_mtt/PathMNIST.stdout.log` | `buffers/official_mtt/COVID/ConvNet/replay_buffer_0.pt` + `results/official_smoke/current_mtt/COVID.stdout.log` | `buffers/official_mtt/Kvasir/ConvNet/replay_buffer_0.pt` + `results/official_smoke/current_mtt/Kvasir.stdout.log` |
| HoP-TM | `buffers/official_smoke/hop_full/PathMNIST_NO_ZCA/ConvNet/replay_buffer_0.pt` + `official_smoke/current_hop/PathMNIST.stdout.log` | `buffers/official_smoke/hop_full/COVID_NO_ZCA/ConvNet/replay_buffer_0.pt` + `official_smoke/current_hop/COVID.stdout.log` | `buffers/official_smoke/hop/Kvasir_NO_ZCA/ConvNet/replay_buffer_0.pt` + `official_smoke/current_hop/Kvasir.stdout.log` |
| NCFM | `results/official_smoke/ncfm_pretrained/pathmnist/premodel0_{init,trained}.pth.tar` + `ncfm/.../data_init.pt` | `results/official_smoke/ncfm_pretrained/covid/premodel0_{init,trained}.pth.tar` + `ncfm/.../data_init.pt` | `results/official_smoke/ncfm_pretrained/kvasir/premodel0_{init,trained}.pth.tar` + `ncfm/.../data_init.pt` |
| DataDAM | `results/smoke/official_DataDAM_PathMNIST_eval/res_DataDAM_PathMNIST_ConvNet_1ipc_.pt` | `results/smoke/official_DataDAM_COVID/res_DataDAM_COVID_ConvNet_1ipc_.pt` | `results/smoke/official_DataDAM_Kvasir/res_DataDAM_Kvasir_ConvNet_1ipc_.pt` |
| CAFE | `results/smoke/official_CAFE_PathMNIST/res_DC_PathMNIST_ConvNet_1ipc.pt` | `results/smoke/official_CAFE_COVID/res_DC_COVID_ConvNet_1ipc.pt` | `results/smoke/official_CAFE_Kvasir/res_DC_Kvasir_ConvNet_1ipc.pt` |

CAFE 的 `res_DC_*` 是原始脚本沿用的文件名前缀，不能据此判断调用了 DC；实际内容来自 CAFE 的 feature alignment。NCFM 表中的 `...` 仅表示其按时间戳生成的深层目录，三个 `data_init.pt` 均已存在。

### 统一 smoke 参数

- `ipc=1`
- `Iteration=0`（NCFM 使用 `niter=1`）
- MTT/HoP-TM buffer：`num_experts=1`、`train_epochs/expert_epochs=1`
- distill：`syn_steps=1`
- 评估型入口通常 `num_exp=1`、`num_eval=1`、`epoch_eval_train=1`
- DM COVID/Kvasir 为了避免单卡 Windows 评估阶段超时，使用 `num_eval=0`，但仍完成真实数据加载、synthetic 初始化、DM 更新和结果保存；不要把这两次 smoke 的输出当准确率。

## 5. 验证命令

```powershell
python -m compileall -q adapted utils scripts test_ncfm_adapted.py
python scripts/validate_medical_adapters.py
python scripts/validate_official_smoke.py
python scripts/run_medical_one_step.py --algorithm DC --dataset PathMNIST
python test_ncfm_adapted.py
```

NCFM 的正确单卡启动方式（工作目录为 `adapted/ncfm`）是：

```powershell
python pretrain/pretrain_script.py --config_path ../../configs/ncfm/pathmnist/ipc1_smoke.yaml --run_mode Pretrain --gpu 0
python condense/condense_script.py --config_path ../../configs/ncfm/pathmnist/ipc1_smoke.yaml --run_mode Condense --gpu 0 -i 1 --init mix
```

one-step 矩阵由下面的单组合命令覆盖全部 24 个组合；其中 MTT/HoP-TM/NCFM 是快速网络/轨迹探针，不能替代它们的 buffer/condense 主流程：

```powershell
python scripts/run_medical_one_step.py --algorithm <DC|DSA|DM|MTT|HoP-TM|NCFM|DataDAM|CAFE> --dataset <PathMNIST|COVID|Kvasir>
```

最终结果（2026-08-10 当前工作树复核）：`compileall PASS`、prepared 三套数据验证 PASS、配置路径 `18/18`、loader `24/24`、官方产物审计 `24/24`、当前工作树重新执行的 one-step `24/24`、NCFM `3/3`；当前 MTT/HoP-TM 官方 trajectory distill 各 `3/3`。本次 one-step 的 24 个 loss/更新均返回 PASS，覆盖每个算法和每个数据集。

## 6. 修改边界与 Git

- `raw/CAFE`、`raw/DataDAM`、`raw/DatasetCondensation`、`raw/HoP-TM`、`raw/mtt-distillation`、`raw/NCFM` 六个官方嵌套仓库保持 clean。
- 所有适配代码位于 `adapted/`；`raw/` 仅作为官方原始参考和版本回溯来源。
- `configs/`、`scripts/`、`utils/`、`docs/` 是外层 benchmark 的新增/修改内容。
- `data/raw`、`data/prepared` 的大文件、`buffers/`、`results/`、`wandb/`、`logged_files/` 都不应进入源码提交。
- 外层 Git 当前已有 checkpoint；完成本报告后应再提交一次最终适配 checkpoint，提交前确认 `git status` 只包含预期源码/文档，且六个 `raw/*` 仓库 clean。

## 7. 不应作出的结论

本报告证明的是：三套数据能被各算法正确读取，核心更新能执行，官方最小入口能保存产物，且输入/输出合同一致。它不证明：

1. IPC10/50 的完整训练已经完成；
2. smoke 的准确率可以作为论文结果；
3. 三个数据集的预处理统计量都是官方发布值；
4. Windows 单卡性能代表 Linux 多卡正式实验性能。

正式实验应从 `configs/*/*/*_full.yaml` 开始，保存完整命令、commit hash、随机种子、环境版本和独立评估结果。
