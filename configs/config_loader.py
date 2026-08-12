#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件加载和管理工具
用于统一加载YAML配置文件并转换为命令行参数
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List


class ConfigLoader:
    """配置文件加载器"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def to_args(self, flat: bool = True) -> Dict[str, Any]:
        """
        将配置转换为参数字典

        Args:
            flat: 是否展平嵌套字典

        Returns:
            参数字典
        """
        if flat:
            return self._flatten_dict(self.config)
        return self.config

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """展平嵌套字典"""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                nested = self._flatten_dict(v, new_key, sep=sep)
                for nested_key, nested_value in nested.items():
                    if nested_key in items:
                        raise ValueError(
                            f"配置展开后出现重复参数名 '{nested_key}'，请重命名字段后再运行。"
                        )
                    items[nested_key] = nested_value
            else:
                if new_key in items:
                    raise ValueError(
                        f"配置展开后出现重复参数名 '{new_key}'，请重命名字段后再运行。"
                    )
                items[new_key] = v
        return items

    def to_argparse(self) -> argparse.Namespace:
        """转换为argparse.Namespace对象"""
        args_dict = self.to_args(flat=True)
        return argparse.Namespace(**args_dict)

    def to_cmd_args(self) -> List[str]:
        """
        转换为命令行参数列表

        Returns:
            命令行参数列表，例如: ['--dataset', 'PathMNIST', '--ipc', '10']
        """
        args_dict = self.to_args(flat=True)
        cmd_args = []

        for key, value in args_dict.items():
            # 跳过嵌套配置和None值
            if isinstance(value, dict) or value is None:
                continue

            # 转换为命令行参数格式
            arg_name = f"--{key}"

            # 处理布尔值
            if isinstance(value, bool):
                if value:
                    cmd_args.append(arg_name)
            # 处理列表
            elif isinstance(value, list):
                cmd_args.append(arg_name)
                cmd_args.extend([str(v) for v in value])
            # 处理其他类型
            else:
                cmd_args.append(arg_name)
                cmd_args.append(str(value))

        return cmd_args

    def merge_with_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        将配置文件与命令行参数合并（命令行参数优先）

        Args:
            args: 命令行参数

        Returns:
            合并后的参数
        """
        config_args = self.to_argparse()

        # 命令行参数覆盖配置文件
        for key, value in vars(args).items():
            if value is not None:
                setattr(config_args, key, value)

        return config_args

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def update(self, key: str, value: Any):
        """更新配置项"""
        keys = key.split('.')
        d = self.config

        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]

        d[keys[-1]] = value

    def save(self, output_path: str = None):
        """保存配置到文件"""
        if output_path is None:
            output_path = self.config_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)


def load_config(config_path: str) -> ConfigLoader:
    """快捷函数：加载配置文件"""
    return ConfigLoader(config_path)


def get_config_path(algorithm: str, dataset: str, ipc: int = 10,
                     config_type: str = 'full') -> Path:
    """
    获取配置文件路径

    Args:
        algorithm: 算法名称 (dc_dsa_dm, hop_tm, mtt, ncfm, datadam, cafe)
        dataset: 数据集名称 (pathmnist, covid, kvasir)
        ipc: Images Per Class
        config_type: 配置类型 (quick, full, best)

    Returns:
        配置文件路径
    """
    base_dir = Path(__file__).parent
    dataset_lower = dataset.lower()

    # 所有当前适配算法都按 dataset 子目录组织配置；NCFM 也使用同一层级，
    # 不能按旧版的 configs/ncfm/<dataset>.yaml 推导路径。
    if algorithm in {'dc', 'dsa', 'dm'}:
        # 三个方法共用 adapted 目录，但必须保留各自独立的参数文件。
        filename = f"ipc{ipc}_{algorithm}_{config_type}.yaml"
        algorithm = 'dc_dsa_dm'
    elif algorithm == 'dc_dsa_dm':
        filename = f"ipc{ipc}_dc_{config_type}.yaml"
    else:
        filename = f"ipc{ipc}_{config_type}.yaml"
    config_path = base_dir / algorithm / dataset_lower / filename

    return config_path


def list_available_configs(algorithm: str = None) -> Dict[str, List[str]]:
    """
    列出所有可用的配置文件

    Args:
        algorithm: 算法名称（None表示列出所有）

    Returns:
        {算法名: [配置文件路径]}
    """
    base_dir = Path(__file__).parent
    configs = {}

    if algorithm:
        algorithms = [algorithm]
    else:
        algorithms = ['dc_dsa_dm', 'hop_tm', 'mtt', 'ncfm', 'datadam', 'cafe']

    for algo in algorithms:
        algo_dir = base_dir / algo
        if algo_dir.exists():
            yaml_files = list(algo_dir.rglob("*.yaml"))
            configs[algo] = [str(f.relative_to(base_dir)) for f in yaml_files]

    return configs


def print_config(config_path: str):
    """打印配置文件内容"""
    loader = ConfigLoader(config_path)

    print(f"\n{'='*80}")
    print(f"配置文件: {config_path}")
    print(f"{'='*80}\n")

    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print_dict(loader.config)
    print()


# 使用示例
if __name__ == "__main__":
    # 示例1: 加载配置文件
    print("示例1: 加载配置文件")
    config = load_config("dc_dsa_dm/pathmnist/ipc10_dc_full.yaml")
    print(f"Dataset: {config.get('dataset')}")
    print(f"IPC: {config.get('ipc')}")
    print(f"Iteration: {config.get('Iteration')}")

    # 示例2: 转换为命令行参数
    print("\n示例2: 转换为命令行参数")
    cmd_args = config.to_cmd_args()
    print(f"命令行参数: {' '.join(cmd_args[:10])}...")

    # 示例3: 列出所有配置
    print("\n示例3: 列出所有配置")
    all_configs = list_available_configs()
    for algo, configs in all_configs.items():
        print(f"\n{algo}:")
        for cfg in configs[:3]:  # 只显示前3个
            print(f"  - {cfg}")

    # 示例4: 获取配置文件路径
    print("\n示例4: 获取配置文件路径")
    config_path = get_config_path('dc_dsa_dm', 'PathMNIST', ipc=10, config_type='full')
    print(f"配置路径: {config_path}")

    # 示例5: 命令行参数与配置文件合并
    print("\n示例5: 命令行参数与配置文件合并")
    # 模拟命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipc', type=int, default=None)
    parser.add_argument('--Iteration', type=int, default=None)
    cmd_args = parser.parse_args(['--ipc', '50'])

    merged_args = config.merge_with_args(cmd_args)
    print(f"合并后的IPC: {merged_args.ipc} (命令行覆盖)")
    print(f"合并后的Iteration: {merged_args.Iteration} (来自配置文件)")
