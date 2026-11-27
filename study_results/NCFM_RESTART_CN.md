# NCFM 历史结果与重跑

## 可复用的结果

| 实验 | 核验结果 | 用途 |
|---|---|---|
| 11 月 PathMNIST IPC10 原评估 | off 49.97%，overall 47.44%，rival 48.59% | 学生训练漏了归一化，保留用于诊断 |
| 同一批合成集修正复评 | off 80.85 ± 0.26%，overall 80.62 ± 0.10%，rival 80.75 ± 0.13% | 三个蒸馏种子，每份五个学生，验证集选模 |
| 归一化 A/B | 历史 26479_0 50.42 → 83.01%，本轮 off/seed0 47.54 → 80.80% | 同产物、同评估器，两个产物各五次 |
| 8 月同批 HPC 单种子 | NCFM 78.80%，GOIS 79.04%，OC-HM-NCFD 78.93%，HM-LCF 78.76%，CS2 75.87%，Copula-JCF 75.26% | 当时的 HoP 评估协议，原样归档 |

11 月配置、逐种子指标、教师诊断和 Slurm 日志位置见 [复评归档](margin_20261126/normalization_reeval/)。8 月指标来自 HPC 的 `/project/prj-sis01/xuxiaoyu/NCFM_Research/results/variants/` 内各方法的 `slurm-26479_*/metrics.json`，该批使用历史测试选模。78.09% 的旧记录对应另一份产物 `26128_0`。

## 继续投入的顺序

| 优先 | 方法 | 最小验证 |
|---|---|---|
| 基线 | NCFM PathMNIST 适配与 IID 4096 | 预训练新教师，固定同一教师库重跑三个蒸馏种子。采用验证选模，合成解码视图归一化一次。旧文档的 fixed-unique U4096 须按代码核对是每步抽 4096 个独立频率，还是跨步固定同一频率库。|
| 1 | GOIS 或方差缩减 | GOIS 历史只有 +0.245 pp 单种子信号。先比较同计算预算的均匀抽样、重要性抽样与 IID 4096，测真实合成像素梯度 RMSE 与每步耗时。HPC Taylor2-CV 在 256 频率的 RMSE 约 1.35，低于 IID 256 的 3.50，但高于 IID 4096 的 0.90，且该测量单步更慢。先验证实际吞吐，再做完整蒸馏。|
| 2 | 类别竞争间隔的方向和剂量 | 留出教师的竞争类 q10 误差 0.02092 → 0.01587，最终 ACC −0.10 pp。先在现有画布测辅助梯度与校准梯度的夹角和范数；若有冲突，再比较 off、原权重 100、只惩罚过大低分位间隔的单边版本，用少量梯度范数比定剂量。 |
| 3 | CATS early-only | 历史线索主要在 Kvasir。检验相同辅助梯度剂量的早期与后期窗口，要求跨蒸馏种子一致。 |
| 暂停 | F2 Sobol、Cross-Fitted Studentized、F5、A1T、Real-support cycling | 历史交接中 F2 最主要的收益相对 release，Cross-Fit 未超过强基线，F5 是方向代理，A1T 种子反向，cycling 未胜随机覆盖。当前检索未定位这几项完整的逐种子原始证据；它们不能与已核验结果合并。 |

GOIS 单种子记录与旧版本位于 `NCFM_Research/results/variants/gois/` 和 `NCFM_Research/methods/NCFM_METHOD_VARIANTS_20260823/`。梯度诊断来自 `NCFM_Research/experiments/M1_5_variance_reduction/results/`。CATS 的参照设计见本地 NCFM 归档中的 `NCFM_STAGE_ADAPTIVE_AUX_20260803/METHOD_CONTRACT.md`。

## 新运行目录

唯一入口为 [ncfm_fresh_pathmnist.sbatch](../scripts/ncfm_fresh_pathmnist.sbatch)。使用 [作者 NCFM](https://github.com/gszfwsb/NCFM/tree/9e3e60b855fa918337013c8c3d460601690eb58e) 的 CF、校准和解码主体，以及当前 PathMNIST 配置。实验输入为同一份真实 train/val/test 文件。每次提交以 Slurm 作业号创建空目录，写出数据、教师和合成集哈希。

短测使用 1 个教师、2 个预训练 epoch、2 个蒸馏步和 1 个训练两轮的学生，检查完整数据流和评估 JSON。正式重跑用新训练的 20 个教师（种子 7341，每个 60 epoch），三份 IPC10 合成集（种子 7341、7342、7343，每份 20,000 步），每份以五个学生评估 2,000 epoch。沿用 11 月实验的归一化统计和 batch_real=128，使新基线有直接参考。新教师库和 Python 随机种子改变，分数以新日志为准。

重跑的成功判据为 20 对初始化与最终权重、三份 `data_20000.pt`、每份五条基于验证集选模的测试记录，且 7,180 张测试图的混淆矩阵能复算 ACC/BACC。候选调参用训练或验证证据选择，正式测试使用冻结配置。

## 运行记录

HPC 根目录为 `/project/prj-sis01/xuxiaoyu/NCFM_Research/experiments/restart_20261127`，部署代码为 `56354fe`，批处理脚本 SHA256 为 `e5f5862ce17bcbb95073b58ee8857a943cfde3843c46959ce9cdbdc9579a3e6b`。由 `xiaoyuxu2` 在 `gpu-h100` 和 `qos-high-gpu` 提交，每项 1 GPU、8 CPU、32G。

| 作业 | 当前状态 | 日志 | 产物 |
|---|---|---|---|
| 206491 | 排队等待 GPU | `logs/smoke-206491.{out,err}` | `outputs/smoke-206491/` |
| 206492 | 等待短测成功 | `logs/full-206492.{out,err}` | `outputs/full-206492/` |

```bash
ssh luhpc
squeue -j 206491,206492
sacct -X -j 206491,206492 -o JobID,State,Elapsed,ExitCode,NodeList
```

本地已清理 13 个 Python 字节码缓存。HPC 已清理旧复评的 12 个缓存目录和两份可再生的传输包，释放约 739 MB，且保留 12 份评估 JSON、九份已校验的原始画布、运行日志和历史教师权重。
