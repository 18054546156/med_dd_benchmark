#!/usr/bin/env python3
"""参数生效性审计工具

检查配置文件中定义的参数是否真正被代码使用。

检查项：
1. YAML 定义但 CLI 未传递的参数
2. YAML 定义但代码未使用的参数
3. 代码硬编码覆盖配置的参数
4. 不同算法间参数名冲突
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

# 参数审计报告需要在 Windows 上可读、可重定向，不能依赖系统 GBK 编码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_yaml_keys(yaml_path: Path) -> Dict[str, Set[str]]:
    """
    加载 YAML 文件的所有键（包括嵌套）

    Returns:
        {section: {key1, key2, ...}}
    """
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    keys = {"_root": set()}

    def flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    if k not in keys:
                        keys[k] = set()
                    keys[k].add(full_key)
                    flatten(v, full_key)
                else:
                    # 只记录叶子键；section 本身不是需要转发的参数。
                    keys["_root"].add(full_key)

    flatten(config)
    return keys


def find_cli_args(script_path: Path) -> Set[str]:
    """
    查找脚本中定义的 CLI 参数

    Returns:
        {arg_name1, arg_name2, ...}
    """
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    result: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                value = arg.value
                if value.startswith("--"):
                    result.add(value[2:])
    return result


def find_args_usage(script_path: Path) -> Set[str]:
    """
    查找脚本中实际使用的 args.xxx

    Returns:
        {param_name1, param_name2, ...}
    """
    with script_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # 匹配 args.param_name
    pattern = r"args\.([a-zA-Z_][a-zA-Z0-9_]*)"
    matches = re.findall(pattern, content)

    return set(matches)


def check_run_config_forwarding(
    run_config_path: Path,
    algorithm: str
) -> Tuple[Set[str], Set[str]]:
    """
    检查 run_config.py 为特定算法转发了哪些参数

    Returns:
        (forwarded_keys, all_possible_keys)
    """
    with run_config_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # DC/DSA/DM 共用一个 ``if algorithm in {...}`` 分支，不能按
    # ``if algorithm ==`` 查找，否则审计器会错误报告三者没有转发参数。
    if algorithm in {"dc", "dsa", "dm"}:
        pattern = r'if algorithm in \{"dc", "dsa", "dm", "dc_dsa_dm"\}.*?return command'
    else:
        pattern = rf'if algorithm == "{algorithm}".*?return command'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return set(), set()

    block = match.group(0)

    # 提取 add/add_bool(command, "key", ...)，布尔开关同样是生效参数。
    forwarded = set(re.findall(r'(?:add|add_bool)\(command, ["\']([^"\']+)["\']', block))

    # 提取 for key in (...)
    for_keys_matches = re.finditer(r'for key in \((.*?)\):', block, re.DOTALL)
    for for_keys_match in for_keys_matches:
        keys_str = for_keys_match.group(1)
        keys = re.findall(r'["\']([^"\']+)["\']', keys_str)
        forwarded.update(keys)

    return forwarded, set()


# 这些字段是统一数据合同或记录信息，不是某个原始入口的 CLI 参数。
# 它们仍应在 YAML 中保留，但不能被审计器误判成“丢失的训练参数”。
CONTRACT_KEYS = {
    "dataset", "method", "num_classes", "channel", "im_size", "model",
    "data_path", "save_path", "device", "seed", "protocol", "protocol_version",
    "protocol_date", "note", "buffer_path", "project",
}

# 这些字段是说明或资源记录，原始入口没有对应 CLI；保留在 YAML 便于审计，
# 但不应被误报为“配置未生效”。真正的算法参数仍必须落到 CLI 或入口 YAML。
RECORD_ONLY_KEYS = {
    "feature_layers", "attn_type", "match_type", "extract_attention",
    "depth", "width", "norm", "normalization", "momentum", "weight_decay",
    "init", "zca", "seed", "device", "channel", "num_classes", "im_size",
}


def config_path_for(root: Path, algorithm: str, dataset: str) -> Path:
    """返回算法真实配置目录；DC/DSA/DM 共用 dc_dsa_dm 目录。"""
    if algorithm in {"dc", "dsa", "dm"}:
        return root / f"configs/dc_dsa_dm/{dataset.lower()}/ipc10_{algorithm}_full.yaml"
    return root / f"configs/{algorithm}/{dataset.lower()}/ipc10_full.yaml"


def entry_scripts_for(algorithm: str, root: Path) -> list[Path]:
    """返回一个算法所有实际阶段入口，避免只审计 buffer 或 distill 之一。"""
    if algorithm == "mtt":
        return [
            root / "adapted/mtt/buffer.py",
            root / "adapted/mtt/distill.py",
            root / "adapted/mtt/utils.py",
        ]
    if algorithm == "ncfm":
        return [
            root / "adapted/ncfm/pretrain/pretrain_script.py",
            root / "adapted/ncfm/condense/condense_script.py",
            root / "adapted/ncfm/evaluation/evaluation_script.py",
            root / "adapted/ncfm/utils/utils.py",
            root / "adapted/ncfm/utils/init_script.py",
            root / "adapted/ncfm/utils/diffaug.py",
            root / "adapted/ncfm/utils/train_val.py",
            root / "adapted/ncfm/condenser/Condenser.py",
            root / "adapted/ncfm/condenser/evaluate.py",
        ]
    return [
        {
            "cafe": [root / "adapted/cafe/distill.py", root / "adapted/cafe/utils.py"],
            "datadam": [root / "adapted/datadam/main_DataDAM.py", root / "adapted/datadam/utils.py"],
            "hop_tm": root / "adapted/hop_tm/distill/distill_high_order_spl.py",
            "dc": [root / "adapted/dc_dsa_dm/main.py", root / "adapted/dc_dsa_dm/utils.py"],
            "dsa": [root / "adapted/dc_dsa_dm/main.py", root / "adapted/dc_dsa_dm/utils.py"],
            "dm": [root / "adapted/dc_dsa_dm/main_DM.py", root / "adapted/dc_dsa_dm/utils.py"],
        }[algorithm]
    ]


def loaded_by_config_file(algorithm: str) -> bool:
    """判断算法是否在入口内加载完整 YAML，而不是依赖 run_config 逐项转发。"""
    # NCFM 的 ArgsProcessor 和 HoP-TM 的 yacs 配置都会读取 YAML 全部字段。
    return algorithm in {"ncfm", "hop_tm"}


def audit_algorithm(
    config_path: Path,
    algorithm: str,
    entry_script: Path | list[Path],
    run_config_path: Path,
    project_root: Path
) -> Dict[str, any]:
    """
    审计单个算法的参数生效性

    Returns:
        审计报告字典
    """
    report = {
        "algorithm": algorithm,
        "config": str(config_path),
        "entry_script": str(entry_script),
        "issues": []
    }

    # 1. 加载 YAML 定义的参数
    yaml_keys = load_yaml_keys(config_path)
    all_yaml_keys = yaml_keys["_root"]

    # 2. 检查 run_config.py 转发的参数
    forwarded_keys, _ = check_run_config_forwarding(run_config_path, algorithm)

    # 3. 检查入口脚本定义的 CLI 参数
    entry_paths = [entry_script] if isinstance(entry_script, Path) else list(entry_script)
    # 算法映射允许返回单层列表；统一展开一次，避免把 list 当 Path 调用 exists。
    if any(isinstance(path, list) for path in entry_paths):
        entry_paths = [path for group in entry_paths for path in (group if isinstance(group, list) else [group])]
    if all(path.exists() for path in entry_paths):
        cli_args = set().union(*(find_cli_args(path) for path in entry_paths))
        used_args = set().union(*(find_args_usage(path) for path in entry_paths))
    else:
        cli_args = set()
        used_args = set()
        report["issues"].append({
            "type": "missing_entry",
            "severity": "error",
            "message": f"Entry script not found: {entry_script}"
        })

    # 4. 分析问题

    # 问题 A: YAML 定义但 run_config 未转发。
    # NCFM/HoP-TM 会在算法入口内部加载 YAML，因此不应要求统一运行器重复转发。
    if loaded_by_config_file(algorithm):
        not_forwarded = set()
    else:
        normalized_forwarded = {key.split(".")[-1] for key in forwarded_keys}
        not_forwarded = {
            key for key in all_yaml_keys
            if key.split(".")[-1] not in normalized_forwarded
            and key.split(".")[-1] not in CONTRACT_KEYS
            and key.split(".")[-1] not in RECORD_ONLY_KEYS
        }
    if not_forwarded:
        report["issues"].append({
            "type": "yaml_not_forwarded",
            "severity": "warning",
            "params": sorted(not_forwarded),
            "message": f"YAML defines {len(not_forwarded)} params not forwarded by run_config.py"
        })

    # 问题 B: run_config 转发但入口脚本未定义
    # HoP-TM distill 的参数由 cfg 动态注册；其静态 parser 不包含 YAML 字段。
    not_defined = forwarded_keys - cli_args
    # main_DM.py 是固定 DM 入口，不声明 --method；run_config 只把 method
    # 传给共用的 main.py。审计时按真实分支排除这一项。
    if algorithm == "dm":
        not_defined.discard("method")
    if not_defined and all(path.exists() for path in entry_paths) and algorithm not in {"hop_tm", "ncfm"}:
        report["issues"].append({
            "type": "forwarded_not_defined",
            "severity": "error",
            "params": sorted(not_defined),
            "message": f"run_config forwards {len(not_defined)} params not defined in entry script"
        })

    # 问题 C: 入口脚本定义但从未使用
    # 这些是入口控制参数或跨阶段参数，不一定在当前所选阶段的同一文件中
    # 直接出现；它们不能作为算法超参数未生效的证据。
    control_args = {
        "debug", "ipc", "load_path", "run_mode", "tf32", "method", "zca",
        # NCFM 在 Condenser/evaluate.py 等被调用模块中消费这些参数；
        # 入口 parser 本身只是把它们放进 Namespace。
        "aug_type", "kldiv", "softlabel", "temperature", "val_repeat",
        # MTT buffer 与 distill 共用一份适配目录，但 save_path 只由 distill 使用。
        "save_path", "lr_init",
    }
    defined_not_used = cli_args - used_args - control_args
    if defined_not_used:
        report["issues"].append({
            "type": "defined_not_used",
            "severity": "info",
            "params": sorted(defined_not_used),
            "message": f"Entry script defines {len(defined_not_used)} params never used"
        })

    # 统计
    report["stats"] = {
        "yaml_params": len(all_yaml_keys),
        "forwarded_params": len(forwarded_keys),
        "cli_params": len(cli_args),
        "used_params": len(used_args)
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="审计参数生效性")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="项目根目录"
    )
    parser.add_argument(
        "--algorithm",
        choices=["ncfm", "cafe", "datadam", "hop_tm", "mtt", "dc", "dsa", "dm"],
        help="审计特定算法（不指定则审计所有）"
    )
    parser.add_argument(
        "--dataset",
        choices=["PathMNIST", "COVID", "Kvasir"],
        default="COVID",
        help="数据集（用于选择配置文件）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 JSON 报告路径"
    )

    args = parser.parse_args()

    # 算法入口脚本映射
    # 阶段入口由 entry_scripts_for() 统一返回；这里仅保留算法顺序。
    entry_scripts = {
        "ncfm": None, "cafe": None, "datadam": None, "hop_tm": None,
        "mtt": None, "dc": None, "dsa": None, "dm": None,
    }

    run_config = args.root / "scripts/run_config.py"

    # 选择要审计的算法
    if args.algorithm:
        algorithms = [args.algorithm]
    else:
        algorithms = list(entry_scripts.keys())

    all_reports = []

    for algo in algorithms:
        # 查找配置文件
        config_pattern = config_path_for(args.root, algo, args.dataset)
        if not config_pattern.exists():
            print(f"WARNING: Config not found for {algo}/{args.dataset}: {config_pattern}")
            continue

        print(f"\n{'='*60}")
        print(f"Auditing: {algo} / {args.dataset}")
        print(f"{'='*60}")

        report = audit_algorithm(
            config_path=config_pattern,
            algorithm=algo,
        entry_script=entry_scripts_for(algo, args.root),
            run_config_path=run_config,
            project_root=args.root
        )

        all_reports.append(report)

        # 打印报告
        print(f"\nStats:")
        for key, value in report["stats"].items():
            print(f"  {key}: {value}")

        if report["issues"]:
            print(f"\nIssues found: {len(report['issues'])}")
            for issue in report["issues"]:
                severity_symbol = {
                    "error": "❌",
                    "warning": "⚠️",
                    "info": "ℹ️"
                }.get(issue["severity"], "•")
                print(f"\n  {severity_symbol} {issue['type'].upper()} ({issue['severity']})")
                print(f"     {issue['message']}")
                if "params" in issue and len(issue["params"]) <= 10:
                    print(f"     Params: {', '.join(issue['params'])}")
                elif "params" in issue:
                    print(f"     Params: {len(issue['params'])} total (first 10: {', '.join(issue['params'][:10])}...)")
        else:
            print("\n✅ No issues found")

    # 输出 JSON 报告
    if args.output:
        import json
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)
        print(f"\n\nFull report saved to: {args.output}")

    # 汇总
    total_issues = sum(len(r["issues"]) for r in all_reports)
    print(f"\n\n{'='*60}")
    print(f"SUMMARY: Audited {len(all_reports)} algorithms, found {total_issues} issues")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
