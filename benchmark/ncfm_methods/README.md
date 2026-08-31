# NCFM 医学数据复现包

本目录包含四个相互独立的完整工程副本：`ncfm_release`、`m16_feature_map_token`、`m22_token_attention` 和 `dr_ltm`。它们均保留各自的 `condense/`、`condenser/`、`NCFM/`、`pretrain/`、`evaluation/` 和 `config/` 目录，便于在新服务器单独运行。

## 方法与来源

| 目录 | 方法 | 核心改变 |
|---|---|---|
| `ncfm_release` | 原始 NCFM | 神经特征函数的幅度和相位距离 |
| `m16_feature_map_token` | M16 Feature-map Token NCFD | 在特征图层增加 token 统计匹配 |
| `m22_token_attention` | M22 Token Attention | 用 discrepancy 引导 token 权重 |
| `dr_ltm` | DR-LTM | 对局部 token 风险/尾部响应进行匹配 |

## 运行顺序

每个目录都要独立准备环境、数据和教师模型：

```bash
cd benchmark/ncfm_methods/<method>
python pretrain/pretrain.py --config_path config/<dataset>.yaml
python condense/condense_script.py --config_path config/<dataset>.yaml
python evaluation/eval.py --config_path config/<dataset>.yaml
```

不同副本的实际参数名可能不同，以该副本的 `README.md`、`config/` 和 `--help` 为准。推荐先用 PathMNIST IPC=10、seed=0 做 smoke，再运行完整 20K；结果写到仓库外：

```text
results/ncfm_methods/<method>/<dataset>/seed<seed>/
```

不要提交 `data/`、预训练权重、合成数据和日志。新服务器复现时只需 clone 本仓库，再把统一准备好的数据目录映射到配置中的 `data_dir`。

## 已有结果（历史实验）

这些数字来自既有 PathMNIST/KvasirV2 适配实验，不能当作三个数据集、多种子最终结论：

| 方法 | PathMNIST BACC | KvasirV2 BACC |
|---|---:|---:|
| M16 Feature-map Token | 75.40% | 78.38% |
| M22 Token Attention | 75.44% | 76.81% |
| DR-LTM | 76.16% | 79.15% |

完整来源与实验路径见仓库根目录 `docs/innovation_methods_results_paths_20260706.csv`。原始 NCFM 代码只读保存在 `raw/`，本目录是用于复现的独立快照。
