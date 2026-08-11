# 医疗 DD Benchmark 配置

`configs/` 为 8 个数据集蒸馏方法在 3 个医疗数据集上的迁移配置。配置不是
原论文在这些医疗数据集上的官方配置；参数来源和修改边界见：

- [`docs/BENCHMARK_STATUS.md`](../docs/BENCHMARK_STATUS.md)
- [`configs/CONFIG_PARAMETER_POLICY.md`](CONFIG_PARAMETER_POLICY.md)
- [`docs/DATASET_CONTRACT.md`](../docs/DATASET_CONTRACT.md)

## 目录

```text
configs/
├── dc_dsa_dm/{pathmnist,covid,kvasir}/
├── mtt/{pathmnist,covid,kvasir}/
├── hop_tm/{pathmnist,covid,kvasir}/
├── ncfm/{pathmnist,covid,kvasir}/
├── datadam/{pathmnist,covid,kvasir}/
└── cafe/{pathmnist,covid,kvasir}/
```

DC、DSA、DM 共用适配目录，但使用不同入口和不同配置文件。MTT、HoP-TM、
NCFM 具有独立的 buffer 或预训练前置流程。

## 命名

- `ipc1_smoke.yaml`：最小入口验证，不用于正式结果。
- `ipc10_full.yaml`：正式 IPC=10 实验起点。
- `ipc10_dc_full.yaml`、`ipc10_dsa_full.yaml`、`ipc10_dm_full.yaml`：
  DC/DSA/DM 的独立配置。

## 统一入口

```powershell
python scripts/run_config.py --config configs/<algorithm>/<dataset>/<file>.yaml
```

默认只打印解析后的命令；确认配置和路径后再加 `--run`。MTT 和 HoP-TM 必须
分别执行 `--stage buffer`、`--stage distill`。NCFM 必须执行
`--stage pretrain`、`--stage condense`，最后使用合成数据目录执行
`--stage evaluation --load-path <path>`。
