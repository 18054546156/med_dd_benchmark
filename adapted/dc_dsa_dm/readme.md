# DC/DSA/DM 医疗数据集适配版本

## 修改说明

**适配日期**: 2024-08-10  
**原始代码**: `raw/DatasetCondensation/`  
**适配版本**: `adapted/dc_dsa_dm/`

## 修改内容

### 主要修改文件：`utils.py`

添加了三个医疗数据集的支持：

1. **PathMNIST** - 病理组织分类
   - 类别数: 9
   - 图像尺寸: 32×32
   - 通道数: 3 (RGB)
   - 数据源: medmnist库（自动下载）

2. **COVID** - COVID-19 X光片严重程度分类
   - 类别数: 4 (Normal, Mild, Moderate, Severe)
   - 图像尺寸: 112×112
   - 通道数: 3 (RGB)
   - 数据源: ImageFolder格式

3. **Kvasir** - 消化道内窥镜病变分类
   - 类别数: 8
   - 图像尺寸: 128×128
   - 通道数: 3 (RGB)
   - 数据源: ImageFolder格式

## 使用方法

### 运行示例

```bash
# PathMNIST
python main.py --dataset PathMNIST --method DC --ipc 10

# COVID
python main.py --dataset COVID --method DC --ipc 10

# Kvasir
python main.py --dataset Kvasir --method DC --ipc 10
```

## 输出结果

在`result/`目录生成合成数据集和评估结果
