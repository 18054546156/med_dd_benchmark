# 医学数据 Benchmark 运行手册

## 目标

对 `PathMNIST`、`COVID`、`Kvasir` 完成：

```text
真实数据校验
 -> NCFM pretrain(20 teachers) -> condense(IPC10, 20000 iterations)
 -> HoP buffer(100 trajectories) -> distill(10000 iterations)
 -> 同一 controlled evaluator
 -> Phase 1 八项诊断
 -> Phase 2 三个真实 NCFM 变体
 -> 正式报告和图表
```

正式结论只允许来自 HPC 的真实 artifact、日志、配置和 SHA256。旧日志、toy 结果和按修改时间挑出的文件不属于证据。

## 运行

登录 HPC 后：

```bash
cd /project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark
bash scripts/submit_full_real_benchmark.sh
```

提交后会把所有依赖关系和 Slurm job ID 写入：

```text
research/ncfm_medical_analysis/submissions/<RUN_TAG>.env
```

VPN 断开重连后，可先读取这个文件，再用其中的 job ID 执行 `squeue`、
`sacct` 和日志检查。

总控会先提交：

1. train-only `statistics.json`
2. CPU loader contract
3. GPU runtime contract
4. 三个 NCFM 作业，每个 1 节点 8 GPU
5. 三个 HoP 作业，每个 1 节点 4 GPU

只有前置检查成功，算法作业才会启动。六个算法作业全部成功后，才会生成正式 manifest 和后续分析作业。

## 监控

```bash
squeue -u "$USER"
squeue -o "%.18i %.12P %.24j %.10T %.10M %.6D %R"
sinfo -o "%P %a %l %D %t %N"

tail -f logs/ncfm-pipeline-<JOBID>.out
tail -f logs/ncfm-pipeline-<JOBID>.err
tail -f logs/hop-tm-4gpu-<JOBNAME>-<JOBID>.out
tail -f logs/hop-tm-4gpu-<JOBNAME>-<JOBID>.err
```

看到 `FAILED`、`CANCELLED` 或 `DependencyNeverSatisfied` 时，不要直接继续下游实验；先保留日志并使用新的 `RUN_TAG` 重跑失败阶段。

## 关键产物

```text
data/prepared/<dataset>/statistics.json

pretrained_models/ncfm/<dataset>/<RUN_ID>/premodel{0..19}_init.pth.tar
pretrained_models/ncfm/<dataset>/<RUN_ID>/premodel{0..19}_trained.pth.tar

results/ncfm/condense/<slug>/ipc10/*_<RUN_ID>/distilled_data/data_20000.pt
results/hop_tm/<dataset>/ipc10/<RUN_ID>/synthetic_data.pt
buffers/hop_tm/<slug>/<RUN_ID>/<dataset>_NO_ZCA/<model>/replay_buffer_*.pt

research/ncfm_mathematical_analysis/runs/ncfm/<slug>/<RUN_ID>/run_manifest.json
research/ncfm_mathematical_analysis/runs/hop_tm/<slug>/<RUN_ID>/run_manifest.json
research/ncfm_medical_analysis/formal_artifact_manifest.json
research/ncfm_medical_analysis/formal_eval_manifest.json
```

## 公平性协议

NCFM 的 `DSA + CutMix` 和 HoP 的原始 `DSA` 属于各自方法的内部设计，不为表面一致而删除。两种方法生成 synthetic data 后，必须使用同一个 evaluator：相同 split、train-only mean/std、模型架构、优化器、epoch、batch size、seed 和测试指标。

因此报告分成两栏：

- 原始方法复现：保留各自官方内部配置。
- Controlled downstream evaluation：只比较 synthetic data 在同一训练测试协议下的质量。

## Phase 2

Phase 2 真实运行三个变体：

```text
qmc             scrambled Sobol frequency sampler
importance      exact p/q weighting, no clipping
learned_frequency    learned frequency sampler (internal flag: NCFM_SAMPLING_NET=true)
```

三个变体都复用明确指定的 baseline teacher，但使用新的 `RUN_ID`、condense 结果和 evaluator 输出。`phase2_variant_report.json` 中的 delta 只是 paired observation；没有达到预注册重复和门禁要求时，报告必须是 `insufficient_evidence`，不能写成方法有效或缺陷已证实。

## 结论规则

```text
complete             测量流程完成
supported            观察到预注册现象，但不足以证明因果缺陷
confirmed            artifact gate、配对评估、重复次数和 effect rule 全部通过
insufficient_evidence 缺少真实产物、配对结果或独立重复
```

医学任务分析会另外输出 class balance、train/val/test 分布、每个 teacher 的 train/val/test ACC、混淆矩阵和 per-class ACC。它们用于定位任务难度和后续调参，不能替代 synthetic-data controlled evaluation。
