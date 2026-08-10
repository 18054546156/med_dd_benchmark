# 医疗数据集 DD 适配变更报告

## 结论

本次工作完成了三套医疗数据的官方下载/预处理、统一数据合同，以及 8 个 DD 算法的数据入口适配：

- PathMNIST、COVID、Kvasir 的 `8 x 3 = 24` 个 loader 组合全部通过。
- 三个数据集上的 8 个算法一步计算全部通过，共 `24/24`。
- MTT 和 HoP-TM 的官方专家轨迹 buffer 均完成一次最小生成。
- DC、DSA、DM、DataDAM、CAFE 的官方 CLI 最小入口已在 COVID 上产出结果文件。

这些结果证明的是“数据能够穿过各算法原有入口、网络、损失和最小更新”，不等于已经完成论文规模的 condensation/distillation 实验，也不代表三套数据上的最终精度结论。

## Git 边界

外层仓库已建立 Git，初始检查点为：

```text
5574aa5 checkpoint: existing medical adaptation baseline
```

仓库结构中的 `raw/` 包含 6 个官方嵌套仓库：CAFE、DataDAM、DatasetCondensation、HoP-TM、mtt-distillation、NCFM。外层 `.gitignore` 不跟踪 `raw/`，并且本次检查中 6 个嵌套仓库的工作区和 index 都是 clean。所有修改都在 `adapted/`、共享工具、配置、测试和文档中。

需要注意：`adapted/` 不是全部从零编写的代码，而是官方代码副本加医疗适配修改；初始检查点已经包含这些副本。当前 Git 的 `M` 表示检查点之后的增量修改，`??` 表示新增的测试、报告或 manifest，不表示整个文件都是新写的。

初始检查点还包含一部分已准备的 Kvasir 图片文件；当前 `.gitignore` 会阻止后续下载数据、buffer 和 results 继续进入版本库，但不会自动取消历史上已经 tracked 的文件。

## 数据合同

| 数据集 | 官方原始格式 | prepared 格式 | 类别 | 输入张量 | 归一化 |
|---|---|---|---:|---|---|
| PathMNIST | MedMNIST `pathmnist.npz` | MedMNIST 读取 | 9 | `3 x 32 x 32` | `[0.741, 0.533, 0.706] / [0.402, 0.821, 0.407]` |
| COVID | Kaggle ZIP，含 `images/` 和 `masks/` | ImageFolder | 4 | `3 x 112 x 112` | ImageNet mean/std |
| Kvasir | Simula Kvasir v2 ZIP | ImageFolder | 8 | `3 x 128 x 128` | ImageNet mean/std |

共享规范和工具：

- `docs/DATASET_CONTRACT.md`：命名、目录、标签和 batch 合同。
- `utils/medical_dataset_utils.py`：`MEDICAL_DATASET_SPECS`、`scalarize_label`、`MedMNISTWrapper`、MedMNIST root 解析和合同验证。
- PathMNIST 的 `np.ndarray([k])` 标签统一转成 Python `int`；ImageFolder 标签使用目录排序得到的整数索引。
- 所有 batch 标签必须是形状 `[N]`、类型 `torch.long`。

## 官方下载与预处理

入口脚本：`scripts/prepare_medical_data.py`。

官方下载来源：

- PathMNIST：MedMNIST 官方 Zenodo 文件，脚本同时使用 `medmnist.PathMNIST(download=True)` 校验。
- COVID：Kaggle 数据集 `tawsifurrahman/covid19-radiography-database`。
- Kvasir：Simula 官方 `kvasir-dataset-v2.zip`。

命令：

```powershell
python scripts/prepare_medical_data.py --data-root data download --dataset PathMNIST
python scripts/prepare_medical_data.py --data-root data download --dataset COVID
python scripts/prepare_medical_data.py --data-root data download --dataset Kvasir

python scripts/prepare_medical_data.py --data-root data prepare --dataset PathMNIST
python scripts/prepare_medical_data.py --data-root data prepare --dataset COVID --source-dir <COVID-19_Radiography_Dataset>
python scripts/prepare_medical_data.py --data-root data prepare --dataset Kvasir --source-dir <kvasir-dataset-v2>
```

下载结果：

```text
data/raw/PathMNIST/pathmnist.npz
data/raw/COVID/covid19-radiography-database.zip
data/raw/Kvasir/kvasir-dataset-v2.zip
```

准备结果：

```text
data/prepared/PathMNIST/pathmnist.npz
data/prepared/COVID/{train,test}/{COVID,Lung_Opacity,Normal,Viral_Pneumonia}/
data/prepared/Kvasir/{train,test}/{8 classes}/
```

COVID 只使用官方 `images/`，不会把 `masks/` 当作分类图片。prepared 阶段会统一转 RGB、resize、按类别做确定性 train/test 划分，并写入每个数据集的 `manifest.json`。

已验证的 prepared 数量：

```text
COVID:
  COVID            train 2893  test  723
  Lung_Opacity     train 4810  test 1202
  Normal           train 8154  test 2038
  Viral_Pneumonia  train 1076  test  269
Kvasir: 8 classes, train 800 + test 200 per class
PathMNIST: official MedMNIST NPZ, MD5 A8B06965200029087D5BD730944A56C1
```

## 算法流程和函数合同

### DC / DSA / DM

代码：`adapted/dc_dsa_dm/`。

入口：`main.py --method DC|DSA`，`main_DM.py`。

数据函数：

```text
get_dataset(dataset, data_path)
  -> channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader
get_network(model, channel, num_classes, im_size)
  -> ConvNet
match_loss(gw_syn, gw_real, args)
  -> scalar matching loss
evaluate_synset(it_eval, net, images_train, labels_train, testloader, args)
  -> evaluation statistics
```

流程仍是官方流程：按类读取真实图像，初始化 `image_syn`，随机初始化网络，在真实/合成 batch 上计算分类梯度，DC 使用梯度匹配，DSA 在梯度匹配前加入可微 Siamese augmentation，DM 使用分布匹配路径，最后调用官方评估函数。

医疗修改集中在 `get_dataset` 的三类分支、尺寸/类别元数据和 PathMNIST 标签标量化；没有把 DC、DSA、DM 改成新的共同训练器。另修复了当前 PyTorch 对 Siamese augmentation 中重叠内存赋值的检查，所有 `tensor[:] = tensor[0]` 改为等价的 `tensor[:] = tensor[0].clone()`。

### MTT

代码：`adapted/mtt/`。

数据函数：

```text
get_dataset(dataset, data_path, batch_size=1, subset="imagenette", args=None)
  -> channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader, loader_train_dict,
     class_map, class_map_inv
```

官方流程分两步：

1. `buffer.py` 用真实训练集训练随机 teacher，保存每个 expert 的参数时间戳轨迹。
2. `distill.py` 读取 replay buffer，沿轨迹采样起点和终点，更新合成图像使学生轨迹接近 expert 轨迹。

医疗适配保留 `loader_train_dict` 的逐类结构，并用 `targets/labels` 元数据构造类别 loader，避免为分类索引反复解码所有图片。active augmentation 的 batch-wise 赋值也做了 `.clone()` 兼容修复。

buffer 目录格式：

```text
buffers/<dataset>/ConvNet/replay_buffer_0.pt
```

### HoP-TM

代码：`adapted/hop_tm/`。

数据函数（baseline 和 GSAM 都保持各自签名）：

```text
get_dataset(dataset, data_path, batch_size=1, subset="imagenette", args=None, baseline=False)
  -> channel, im_size, num_classes, mean, std,
     dst_train, dst_test, testloader, loader_train_dict,
     class_map, class_map_inv

get_dataset(dataset, data_path, batch_size=1, args=None)  # GSAM
  -> 同样的 11 项返回值
```

官方流程是 FTD/GSAM buffer 生成，再由 `DATM.py`、`DATM_cal_time.py`、`DATM_tesla.py` 或 `distill_high_order_spl.py` 读取 expert trajectory，进行高阶轨迹匹配。医疗修改包括 `PathMnist`/`PathMNIST`/`COVID`/`Kvasir` 的 `_NO_ZCA` 路径命名，以及 baseline/GSAM 两套 loader 和 buffer utility 的数据分支。

buffer 目录格式：

```text
buffers/<dataset>_NO_ZCA/ConvNet/replay_buffer_0.pt
```

### DataDAM

代码：`adapted/datadam/`。

数据函数：

```text
get_dataset(dataset, data_path, args)
  -> channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader, zca
get_network(model, channel, num_classes, im_size)
  -> DataDAM ConvNet
evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, skip=False)
  -> evaluation statistics
```

官方流程是注意力/特征匹配和分类输出匹配共同更新 `image_syn`。医疗修改是 loader、类别/尺寸元数据和 PathMNIST 标签标量化；同时修复 attention loss 中 `.cpu()` 截断 `image_syn` 梯度的问题，并让初始 loss tensor 使用当前设备。

### CAFE

代码：`adapted/cafe/`。

数据函数：

```text
get_dataset(dataset, data_path)
  -> channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_test, testloader
get_network(model, channel, num_classes, im_size)
  -> CAFE ConvNet
match_loss(gw_syn, gw_real, args)
  -> scalar gradient loss
evaluate_synset(it_eval, net, images_train, labels_train, testloader, args)
  -> evaluation statistics
```

官方流程是中间特征对齐、分类输出和梯度匹配的联合损失。`distill.py` 中原本固定 CIFAR-10 的 `10` 类和 `3x32x32` reshape，已经改为使用 `num_classes` 和真实 batch 的动态尺寸；合成/真实图像拼接也改为 `torch.cat`，保留医疗数据尺寸。

### NCFM

代码：`adapted/ncfm/`。

数据函数：

```text
load_resized_data(dataset, data_dir, size, nclass, load_memory, seed)
  -> train_dataset, val_dataset

get_loader(args), args.run_mode == "Condense"
  -> ClassDataLoader(train_set), None
get_loader(args), args.run_mode == "Evaluation"
  -> None, DataLoader(val_dataset)
get_loader(args), args.run_mode == "Pretrain"
  -> train_loader, val_loader, train_sampler
```

`ClassDataLoader` 和 `ClassMemDataLoader` 仍是 NCFM 原有类别采样接口；医疗适配只补齐 `nclass`、一维整数 `targets`、路径解析和 CPU fallback。`ClassMemDataLoader` 会把数据移到指定 `args.device`，不再无条件调用 CUDA。

## 配置文件

配置按算法原有参数组织，不共享一套强行统一的训练参数：

```text
configs/dc_dsa_dm/{pathmnist,covid,kvasir}/
configs/mtt/{pathmnist,covid,kvasir}/
configs/hop_tm/{pathmnist,covid,kvasir}/
configs/datadam/{pathmnist,covid,kvasir}/
configs/cafe/{pathmnist,covid,kvasir}/
configs/ncfm/{pathmnist,covid,kvasir}/
```

NCFM 的原生配置仍保留在 `adapted/ncfm/config/ipc10/medical/`；`configs/ncfm/` 是从仓库根目录执行时使用的入口副本。MTT 先生成 buffer，HoP-TM 先生成 FTD/GSAM buffer；其他算法直接从各自 CLI 参数启动。

## 验证命令和结果

语法检查：

```powershell
python -m py_compile adapted/ncfm/utils/utils.py adapted/ncfm/data/dataloader.py
python -m compileall -q adapted utils scripts test_ncfm_adapted.py
```

24 个 loader：

```powershell
python scripts/validate_medical_adapters.py
```

结果：`SUMMARY passed=24/24 failed=0`。

一步计算：

```powershell
python scripts/run_medical_one_step.py --algorithm <DC|DSA|DM|MTT|HoP-TM|NCFM|DataDAM|CAFE> --dataset <PathMNIST|COVID|Kvasir>
```

结果：8 个算法在三个数据集上均 PASS，包含梯度匹配、分类反向更新和 MTT/HoP-TM 轨迹写入。

NCFM 完整 loader 链路额外验证：

```text
PathMNIST: ClassDataLoader PASS, ClassMemDataLoader PASS
COVID:     ClassDataLoader PASS, ClassMemDataLoader PASS
Kvasir:    ClassDataLoader PASS, ClassMemDataLoader PASS
```

官方最小 CLI / buffer：

```text
DC       COVID Iteration=0 结果文件 PASS
DSA      COVID Iteration=0 结果文件 PASS
DM       COVID Iteration=0 结果文件 PASS
DataDAM  COVID Iteration=0, num_eval=1 结果文件 PASS
CAFE     COVID Iteration=0, num_eval=1 结果文件已生成
MTT      Kvasir 1 expert x 1 epoch replay_buffer_0.pt PASS
HoP-TM   Kvasir 1 expert x 1 epoch FTD replay_buffer_0.pt PASS
```

生成的关键文件位于被忽略的 `buffers/official_smoke/` 和 `results/official_smoke/`。`num_eval=0` 对 DataDAM 的官方收尾代码不合法，会触发 `np.max([])`；正式 smoke 使用 `num_eval=1`。CAFE 的外层 shell 在 600 秒达到超时，但其子进程随后完成并生成了结果文件，因此这里只把它记为“产物级 CLI 通过”，不把耗时当作性能结论。

## 尚未完成的实验工作

以下项目仍需在正式 GPU 环境和论文规模参数下单独运行：

- 三个数据集、8 个算法的完整 condensation/distillation 迭代。
- MTT 和 HoP-TM 的正式数量 expert buffer。
- NCFM 的 pretrain、condense、evaluation 全流程。
- 正式精度、显存、耗时和多随机种子统计。

因此当前准确状态是：数据下载/预处理完成，算法 loader 和最小计算链路完成，正式 benchmark 结果尚未声称完成。
