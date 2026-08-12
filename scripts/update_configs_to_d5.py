#!/usr/bin/env python3
"""批量修改配置文件，将 COVID 和 Kvasir 统一改成 D5

原则：
- PathMNIST: 保持 D3（已经正确）
- COVID: 统一改成 D5（跟随 HoP-TM）
- Kvasir: 统一改成 D5（参考 COVID）
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

def update_dc_dsa_dm(dataset: str):
    """更新 DC/DSA/DM 配置 (D3 → D5)"""
    config_dir = ROOT / f"configs/dc_dsa_dm/{dataset}"

    if not config_dir.exists():
        print(f"[WARN]  Directory not found: {config_dir}")
        return

    for config_file in config_dir.glob("*.yaml"):
        content = config_file.read_text(encoding="utf-8")
        original = content

        # model: ConvNet → model: ConvNetD5
        # 只替换没有明确深度的 ConvNet
        content = re.sub(
            r'^model: ConvNet\s*$',
            'model: ConvNetD5',
            content,
            flags=re.MULTILINE
        )

        # 如果是 ConvNetD3，也改成 ConvNetD5
        content = re.sub(
            r'ConvNetD3',
            'ConvNetD5',
            content
        )

        if content != original:
            config_file.write_text(content, encoding="utf-8")
            print(f"[OK] Updated: {config_file.relative_to(ROOT)}")
        else:
            print(f"     No change: {config_file.relative_to(ROOT)}")

def update_mtt(dataset: str):
    """更新 MTT 配置 (D4 → D5)"""
    config_file = ROOT / f"configs/mtt/{dataset}/ipc10_full.yaml"

    if not config_file.exists():
        print(f"[WARN]  File not found: {config_file}")
        return

    content = config_file.read_text(encoding="utf-8")
    original = content

    # ConvNetD4 → ConvNetD5 (顶层 model 和 buffer.model)
    content = re.sub(r'ConvNetD4', 'ConvNetD5', content)

    if content != original:
        config_file.write_text(content, encoding="utf-8")
        print(f"[OK] Updated: {config_file.relative_to(ROOT)}")
    else:
        print(f"   No change (already D5): {config_file.relative_to(ROOT)}")

def update_ncfm(dataset: str):
    """更新 NCFM 配置 (depth: 4 → depth: 5)"""
    config_file = ROOT / f"configs/ncfm/{dataset}/ipc10_full.yaml"

    if not config_file.exists():
        print(f"[WARN]  File not found: {config_file}")
        return

    content = config_file.read_text(encoding="utf-8")
    original = content

    # network.depth: 4 → depth: 5
    content = re.sub(
        r'(\s+)depth:\s*4',
        r'\1depth: 5',
        content
    )

    if content != original:
        config_file.write_text(content, encoding="utf-8")
        print(f"[OK] Updated: {config_file.relative_to(ROOT)}")
    else:
        print(f"   No change (already depth 5): {config_file.relative_to(ROOT)}")

def update_datadam(dataset: str):
    """更新 DataDAM 配置 (D3 → D5)"""
    config_file = ROOT / f"configs/datadam/{dataset}/ipc10_full.yaml"

    if not config_file.exists():
        print(f"[WARN]  File not found: {config_file}")
        return

    content = config_file.read_text(encoding="utf-8")
    original = content

    # model: ConvNet → model: ConvNetD5
    content = re.sub(
        r'^model: ConvNet\s*$',
        'model: ConvNetD5',
        content,
        flags=re.MULTILINE
    )

    # ConvNetD3 → ConvNetD5
    content = re.sub(r'ConvNetD3', 'ConvNetD5', content)

    if content != original:
        config_file.write_text(content, encoding="utf-8")
        print(f"[OK] Updated: {config_file.relative_to(ROOT)}")
    else:
        print(f"   No change: {config_file.relative_to(ROOT)}")

def update_cafe(dataset: str):
    """更新 CAFE 配置 (D3 → D5)"""
    config_file = ROOT / f"configs/cafe/{dataset}/ipc10_full.yaml"

    if not config_file.exists():
        print(f"[WARN]  File not found: {config_file}")
        return

    content = config_file.read_text(encoding="utf-8")
    original = content

    # model: ConvNet → model: ConvNetD5
    content = re.sub(
        r'^model: ConvNet\s*$',
        'model: ConvNetD5',
        content,
        flags=re.MULTILINE
    )

    # ConvNetD3 → ConvNetD5
    content = re.sub(r'ConvNetD3', 'ConvNetD5', content)

    if content != original:
        config_file.write_text(content, encoding="utf-8")
        print(f"[OK] Updated: {config_file.relative_to(ROOT)}")
    else:
        print(f"   No change: {config_file.relative_to(ROOT)}")

def verify_hop_tm(dataset: str):
    """验证 HoP-TM 配置（应该已经是 D5，不修改）"""
    config_file = ROOT / f"configs/hop_tm/{dataset}/ipc10_full.yaml"

    if not config_file.exists():
        print(f"[WARN]  File not found: {config_file}")
        return

    content = config_file.read_text(encoding="utf-8")

    if "ConvNetD5" in content:
        print(f"[OK] HoP-TM already D5: {config_file.relative_to(ROOT)}")
    else:
        print(f"[WARN]  HoP-TM NOT D5: {config_file.relative_to(ROOT)}")

def main():
    print("="*60)
    print("Updating configs to D5 (following HoP-TM)")
    print("="*60)

    # COVID: 所有算法改成 D5
    print("\n=== COVID (all → D5) ===")
    update_dc_dsa_dm("covid")
    update_mtt("covid")
    update_ncfm("covid")
    update_datadam("covid")
    update_cafe("covid")
    verify_hop_tm("covid")

    # Kvasir: 参考 COVID，也改成 D5
    print("\n=== Kvasir (all → D5, following COVID) ===")
    update_dc_dsa_dm("kvasir")
    update_mtt("kvasir")
    update_ncfm("kvasir")
    update_datadam("kvasir")
    update_cafe("kvasir")
    verify_hop_tm("kvasir")

    # PathMNIST: 不修改，已经都是 D3
    print("\n=== PathMNIST (keep D3, no change) ===")
    print("[OK] PathMNIST configs already correct (D3)")

    print("\n" + "="*60)
    print("[OK] All configs updated to follow HoP-TM")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify changes: git diff configs/")
    print("2. Run validation: python scripts/validate_config.py")
    print("3. Sync to server when VPN is ready")

if __name__ == "__main__":
    main()
