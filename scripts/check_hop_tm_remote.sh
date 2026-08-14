#!/usr/bin/env bash
# 只读检查 HoP-TM 两个数据集的作业、轨迹、蒸馏和 DD evaluation 产物。
set -u

ROOT=/project/prj-sis01/xiaoyu_xu/med_dd_project/dd_benchmark
cd "$ROOT"

echo '=== Slurm jobs ==='
squeue -u "$USER" -o '%.12i %.18j %.2t %.10M %.4D %R'

echo '=== HoP-TM output files ==='
find logs -maxdepth 1 -type f -name 'hop-tm-4gpu-*' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort

echo '=== Formal buffers ==='
for spec in \
  'COVID ConvNetD5 buffers/hop_tm/COVID_NO_ZCA/ConvNetD5' \
  'PathMNIST ConvNet buffers/hop_tm/PathMNIST_NO_ZCA/ConvNet'; do
  set -- $spec
  name=$1
  model=$2
  dir=$3
  echo "[$name / $model] $dir"
  find "$dir" -maxdepth 1 -type f -name 'replay_buffer_*.pt' -printf '%f %s bytes\n' 2>/dev/null | sort -V
done

echo '=== Pipeline stages and errors ==='
grep -HnE 'stage=|complete|Traceback|Error|Exception|FAILED|CANCELLED|mean:|std' \
  logs/hop-tm-4gpu-*.out logs/hop-tm-4gpu-*.err \
  logs/hop-tm-4gpu-*.combined.log 2>/dev/null | tail -200

echo '=== Distilled files ==='
find results/hop_tm -type f \( -name 'images_*.pt' -o -name 'labels_*.pt' -o -name 'lr_*.pt' -o -name 'evaluation.log' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort

echo '=== Buffer tensor validation ==='
"${PYTHON:-/home/xiaoyuxu2/.conda/envs/meddd/bin/python}" - <<'PY'
from pathlib import Path
import torch

root = Path('buffers/hop_tm')
for path in sorted(root.glob('*_NO_ZCA/*/replay_buffer_*.pt')):
    data = torch.load(path, map_location='cpu')
    lengths = {len(item) for item in data}
    print(f'{path}: trajectories={len(data)} states={sorted(lengths)}')
PY
