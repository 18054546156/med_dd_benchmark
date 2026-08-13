#!/usr/bin/env python3
"""实验 Manifest 生成器

在每次实验运行后调用，生成完整的实验配置记录，用于：
1. 可复现性：记录完整的运行参数和环境
2. 可追溯性：关联配置、代码版本、数据、结果
3. 可审计性：验证实验是否符合公平比较协议
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_git_commit() -> Optional[str]:
    """获取当前 Git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_status() -> Optional[str]:
    """获取 Git 工作区状态"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        status = result.stdout.strip()
        return "clean" if not status else "dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def compute_file_hash(path: Path) -> str:
    """计算文件 SHA256 hash"""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_config_hash(config: Dict[str, Any]) -> str:
    """计算配置 JSON 的 hash"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_config(path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_backbone_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取 backbone 相关配置"""
    backbone_keys = [
        "model", "depth", "width", "norm", "activation", "pooling",
        "channel", "im_size", "num_classes"
    ]
    backbone = {}
    for key in backbone_keys:
        if key in config:
            backbone[key] = config[key]
        # 检查嵌套字段
        for section in ["network", "model_config"]:
            if section in config and isinstance(config[section], dict):
                if key in config[section]:
                    backbone[key] = config[section][key]
    return backbone


def generate_experiment_id(
    algorithm: str,
    dataset: str,
    ipc: int,
    seed: int,
    timestamp: Optional[str] = None
) -> str:
    """生成实验唯一 ID"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{algorithm}_{dataset}_ipc{ipc}_seed{seed}_{timestamp}"


def create_manifest(
    config_path: Path,
    algorithm: str,
    dataset: str,
    ipc: int,
    seed: int,
    output_dir: Path,
    checkpoints: Optional[list] = None,
    synthetic_data: Optional[list] = None,
    results: Optional[Dict[str, Any]] = None,
    slurm_job_id: Optional[str] = None,
    gpu_info: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Path:
    """
    创建实验 manifest

    Args:
        config_path: 配置文件路径
        algorithm: 算法名称
        dataset: 数据集名称
        ipc: Images per class
        seed: 随机种子
        output_dir: 输出目录
        checkpoints: checkpoint 文件列表
        synthetic_data: 合成数据文件列表
        results: 实验结果字典
        slurm_job_id: Slurm 作业 ID
        gpu_info: GPU 信息
        start_time: 开始时间 (ISO 8601)
        end_time: 结束时间 (ISO 8601)

    Returns:
        manifest 文件路径
    """
    # 加载配置
    config = load_config(config_path)

    # 生成实验 ID
    experiment_id = generate_experiment_id(algorithm, dataset, ipc, seed)

    # 提取 backbone 配置
    backbone = extract_backbone_config(config)
    backbone_hash = compute_config_hash(backbone)

    # 构建 manifest
    manifest = {
        "experiment_id": experiment_id,
        "algorithm": algorithm,
        "dataset": dataset,
        "protocol": config.get("protocol", "unknown"),
        "protocol_version": config.get("protocol_version", "unknown"),

        "backbone": {
            **backbone,
            "config_hash": backbone_hash
        },

        "data": {
            "split_source": config.get("data_split", {}).get("source", "unknown"),
            "data_path": str(config.get("data_path", "data/prepared")),
            "manifest_path": None,  # 待填充
            "manifest_hash": None,  # 待填充
        },

        "algorithm_config": {
            "ipc": ipc,
            "seed": seed,
            # 其他算法特定参数从 config 中提取
        },

        "evaluation": config.get("evaluation", {}),

        "reproducibility": {
            "seed": seed,
            "git_commit": get_git_commit(),
            "git_status": get_git_status(),
            "config_path": str(config_path.resolve()),
            "config_hash": compute_file_hash(config_path),
            "slurm_job_id": slurm_job_id,
            "gpu": gpu_info,
            "start_time": start_time or datetime.now().isoformat(),
            "end_time": end_time,
        },

        "outputs": {
            "checkpoints": [str(p) for p in (checkpoints or [])],
            "synthetic_data": [str(p) for p in (synthetic_data or [])],
            "results": results or {},
        },

        "created_at": datetime.now().isoformat(),
    }

    # 查找数据集 manifest
    data_root = Path(config.get("data_path", "data/prepared"))
    dataset_manifest = data_root / dataset / "manifest.json"
    if dataset_manifest.exists():
        manifest["data"]["manifest_path"] = str(dataset_manifest.resolve())
        manifest["data"]["manifest_hash"] = compute_file_hash(dataset_manifest)
        # 读取数据集 split 信息
        with dataset_manifest.open("r") as f:
            data_manifest = json.load(f)
            if "splits" in data_manifest:
                manifest["data"]["splits"] = data_manifest["splits"]

    # 保存 manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{experiment_id}_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="生成实验 manifest")
    parser.add_argument("--config", type=Path, required=True, help="配置文件路径")
    parser.add_argument("--algorithm", required=True, help="算法名称")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--ipc", type=int, required=True, help="Images per class")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--checkpoints", nargs="*", help="Checkpoint 文件列表")
    parser.add_argument("--synthetic-data", nargs="*", help="合成数据文件列表")
    parser.add_argument("--results-json", type=Path, help="结果 JSON 文件路径")
    parser.add_argument("--slurm-job-id", help="Slurm 作业 ID")
    parser.add_argument("--gpu", help="GPU 信息")
    parser.add_argument("--start-time", help="开始时间 (ISO 8601)")
    parser.add_argument("--end-time", help="结束时间 (ISO 8601)")

    args = parser.parse_args()

    # 加载结果
    results = None
    if args.results_json and args.results_json.exists():
        with args.results_json.open("r") as f:
            results = json.load(f)

    # 创建 manifest
    manifest_path = create_manifest(
        config_path=args.config,
        algorithm=args.algorithm,
        dataset=args.dataset,
        ipc=args.ipc,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoints=args.checkpoints,
        synthetic_data=args.synthetic_data,
        results=results,
        slurm_job_id=args.slurm_job_id,
        gpu_info=args.gpu,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    print(f"Manifest created: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
