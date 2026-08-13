#!/bin/bash
# 快速验证脚本 - 检查所有7项任务是否完成

echo "========================================"
echo "Fair Comparison Infrastructure Check"
echo "========================================"
echo ""

ROOT="D:/Project/med_dd_benchmark"
cd "$ROOT"

PASS=0
FAIL=0

# Task 1: PathMNIST 数据结构
echo "[1/7] Checking PathMNIST data structure..."
if [ -d "data/prepared/PathMNIST/train" ] && \
   [ -d "data/prepared/PathMNIST/val" ] && \
   [ -d "data/prepared/PathMNIST/test" ]; then
    echo "  ✓ PASS: train/val/test directories exist"
    PASS=$((PASS+1))
else
    echo "  ✗ FAIL: Missing train/val/test directories"
    FAIL=$((FAIL+1))
fi

# Task 2: Backbone 配置
echo "[2/7] Checking backbone configs..."
if [ -f "configs/backbones/fair_pathmnist.yaml" ] && \
   [ -f "configs/backbones/fair_covid.yaml" ] && \
   [ -f "configs/backbones/fair_kvasir.yaml" ]; then
    echo "  ✓ PASS: All 3 backbone configs exist"
    PASS=$((PASS+1))
else
    echo "  ✗ FAIL: Missing backbone configs"
    FAIL=$((FAIL+1))
fi

# Task 3: Manifest 生成脚本
echo "[3/7] Checking manifest script..."
if [ -f "scripts/create_manifest.py" ]; then
    echo "  ✓ PASS: create_manifest.py exists"
    PASS=$((PASS+1))
else
    echo "  ✗ FAIL: create_manifest.py missing"
    FAIL=$((FAIL+1))
fi

# Task 4: 目录重组脚本
echo "[4/7] Checking reorganize script..."
if [ -f "scripts/reorganize_results.py" ]; then
    echo "  ✓ PASS: reorganize_results.py exists"
    PASS=$((PASS+1))
else
    echo "  ✗ FAIL: reorganize_results.py missing"
    FAIL=$((FAIL+1))
fi

# Task 5: 参数审计脚本
echo "[5/7] Checking param audit script..."
if [ -f "scripts/audit_param_effectiveness.py" ]; then
    echo "  ✓ PASS: audit_param_effectiveness.py exists"
    PASS=$((PASS+1))
else
    echo "  ✗ FAIL: audit_param_effectiveness.py missing"
    FAIL=$((FAIL+1))
fi

# Task 6: 配置校验脚本
echo "[6/7] Checking config validation script..."
if [ -f "scripts/validate_config.py" ]; then
    echo "  ✓ PASS: validate_config.py exists"
    PASS=$((PASS+1))
else
    echo "  ✗ FAIL: validate_config.py missing"
    FAIL=$((FAIL+1))
fi

# Task 7: Legacy 标记
echo "[7/7] Checking legacy marking..."
LEGACY_COUNT=$(find results -name "LEGACY_MANIFEST.json" 2>/dev/null | wc -l)
if [ "$LEGACY_COUNT" -gt 0 ]; then
    echo "  ✓ PASS: $LEGACY_COUNT legacy directories marked"
    PASS=$((PASS+1))
else
    echo "  ✗ FAIL: No legacy directories marked"
    FAIL=$((FAIL+1))
fi

echo ""
echo "========================================"
echo "Summary: $PASS passed, $FAIL failed"
echo "========================================"

if [ $FAIL -eq 0 ]; then
    echo "✓ All tasks completed successfully!"
    exit 0
else
    echo "✗ Some tasks incomplete. See above."
    exit 1
fi
