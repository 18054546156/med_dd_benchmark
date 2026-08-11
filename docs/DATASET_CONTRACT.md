# 医疗数据集统一合同

本合同是 DC、DSA、DM、MTT、HoP-TM、NCFM、DataDAM 和 CAFE 共用的数据边界。它只规定数据来源、目录、标签和输入张量，不重写任何算法的 condensation/distillation pipeline。

## 1. 官方来源

| 数据集 | 官方来源 | 下载后的关键文件或目录 |
|---|---|---|
| PathMNIST | [MedMNIST](https://medmnist.com/) / [MedMNIST GitHub](https://github.com/MedMNIST/MedMNIST) | `pathmnist.npz`，由 `medmnist.PathMNIST` 读取 |
| COVID | [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | `COVID-19_Radiography_Dataset/{COVID,Lung_Opacity,Normal,Viral Pneumonia}/images/` |
| Kvasir | [Simula Kvasir v2](https://datasets.simula.no/kvasir/) | `kvasir-dataset-v2/{dyed-lifted-polyps,esophagitis,...}` |

下载、解压和准备统一由 `scripts/prepare_medical_data.py` 执行。例如：

```powershell
python scripts/prepare_medical_data.py --data-root data download --dataset PathMNIST
python scripts/prepare_medical_data.py --data-root data download --dataset COVID
python scripts/prepare_medical_data.py --data-root data download --dataset Kvasir

python scripts/prepare_medical_data.py --data-root data prepare --dataset PathMNIST
python scripts/prepare_medical_data.py --data-root data prepare --dataset COVID --source-dir data/raw/COVID/source/COVID-19_Radiography_Dataset
python scripts/prepare_medical_data.py --data-root data prepare --dataset Kvasir --source-dir data/raw/Kvasir/source/kvasir-dataset-v2
```

## 2. 固定属性

| 数据集 | 通道 | 算法输入尺寸 | 类别数 | 训练集 | 验证集 | 测试集 | 标签格式 |
|---|---:|---:|---:|---:|---:|---:|---|
| PathMNIST | 3 | `3 x 32 x 32` | 9 | 89996 | 10004 | 7180 | Python `int` / batch `torch.long` |
| COVID | 3 | `3 x 112 x 112` | 4 | 14817 | 2116 | 4232 | ImageFolder 类别索引 |
| Kvasir | 3 | `3 x 128 x 128` | 8 | 5600 | 800 | 1600 | ImageFolder 类别索引 |

共享定义位于 `utils/medical_dataset_utils.py` 的 `MEDICAL_DATASET_SPECS`。PathMNIST 的 MedMNIST `ndarray([k])` 必须先标量化；ImageFolder 的类别索引按目录排序得到。

## 3. prepared 目录

```text
data/prepared/
  PathMNIST/pathmnist.npz
  COVID/train/{COVID,Lung_Opacity,Normal,Viral_Pneumonia}/*.png
  COVID/val/{COVID,Lung_Opacity,Normal,Viral_Pneumonia}/*.png
  COVID/test/{COVID,Lung_Opacity,Normal,Viral_Pneumonia}/*.png
  Kvasir/train/{8 class directories}/*
  Kvasir/val/{8 class directories}/*
  Kvasir/test/{8 class directories}/*
```

`manifest.json` 保存来源、随机种子、划分比例、类别计数和文件计数。`COVID` 只使用 `images/` 分类图像，不把同目录的 `masks/` 当作分类样本。

## 4. 算法边界

prepared 数据只负责统一文件布局和可复现划分。各官方算法仍保留自己的 resize、normalize、augmentation、buffer、evaluation 和 config 逻辑：

- DC/DSA/DM、DataDAM、CAFE：直接从真实训练集组织 `images_all/labels_all`，输出 synthetic tensor 和评估结果。
- MTT、HoP-TM：先训练 expert，再保存 replay buffer，随后由 distill 脚本读取 trajectory。
- NCFM：先按 YAML 预训练模型，再由 condenser 读取预训练模型执行 condense/evaluation。

因此 loader 通过、buffer 生成、蒸馏完成和最终准确率是四个不同状态，报告必须分别记录。
