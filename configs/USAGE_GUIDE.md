# 配置使用指南

当前 Benchmark 的唯一状态说明见
[`docs/BENCHMARK_STATUS.md`](../docs/BENCHMARK_STATUS.md)。参数来源政策见
[`CONFIG_PARAMETER_POLICY.md`](CONFIG_PARAMETER_POLICY.md)。不要根据历史报告中
按数据集猜测的学习率或迭代次数覆盖 YAML 当前值。

## 1. 先验证配置，不执行训练

```powershell
python scripts/run_config.py --config configs/dc_dsa_dm/pathmnist/ipc10_dc_full.yaml
```

命令会打印解析后的脚本、工作目录和命令行参数。确认后加 `--run` 执行。

## 2. 单阶段方法

```powershell
python scripts/run_config.py --config configs/dc_dsa_dm/pathmnist/ipc10_dc_full.yaml --run
python scripts/run_config.py --config configs/dc_dsa_dm/pathmnist/ipc10_dsa_full.yaml --run
python scripts/run_config.py --config configs/dc_dsa_dm/pathmnist/ipc10_dm_full.yaml --run
python scripts/run_config.py --config configs/datadam/pathmnist/ipc10_full.yaml --algorithm datadam --run
python scripts/run_config.py --config configs/cafe/pathmnist/ipc10_full.yaml --algorithm cafe --run
```

## 3. MTT

MTT 的 buffer 必须匹配数据集、模型、输入尺寸、归一化和数据划分。

```powershell
python scripts/run_config.py --config configs/mtt/covid/ipc10_full.yaml --algorithm mtt --stage buffer --run
python scripts/run_config.py --config configs/mtt/covid/ipc10_full.yaml --algorithm mtt --stage distill --run
```

## 4. HoP-TM

HoP-TM 同样必须先生成当前医疗数据集专属的 expert buffer。

```powershell
python scripts/run_config.py --config configs/hop_tm/kvasir/ipc10_full.yaml --algorithm hop_tm --stage buffer --run
python scripts/run_config.py --config configs/hop_tm/kvasir/ipc10_full.yaml --algorithm hop_tm --stage distill --run
```

## 5. NCFM

NCFM 不是单个蒸馏命令，必须分阶段执行。evaluation 必须显式提供合成数据
目录，防止误评估旧结果。

```powershell
python scripts/run_config.py --config configs/ncfm/pathmnist/ipc10_full.yaml --algorithm ncfm --stage pretrain --run
python scripts/run_config.py --config configs/ncfm/pathmnist/ipc10_full.yaml --algorithm ncfm --stage condense --run
python scripts/run_config.py --config configs/ncfm/pathmnist/ipc10_full.yaml --algorithm ncfm --stage evaluation --load-path <合成数据目录> --run
```

## 6. Smoke 与正式实验

`ipc1_smoke.yaml` 只检查入口、数据合同、前向/反向和文件输出。它不用于比较
算法性能。正式实验从对应的 `ipc10_full.yaml` 开始，并记录命令、Git commit、
随机种子、manifest hash、模型和独立 test 结果。
