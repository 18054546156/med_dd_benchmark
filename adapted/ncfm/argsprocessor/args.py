import argparse
from typing import Dict, Any, Optional
import yaml


class ArgsProcessor:
    def __init__(self, config_path: str) -> None:
        """
        Initialize ArgsProcessor with a configuration file path.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            None
        """
        self.config_path: str = config_path

    def flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Recursively flattens a nested dictionary, but does not add the parent key.
        
        Args:
            d (Dict[str, Any]): Input dictionary to flatten
            parent_key (str, optional): Parent key (unused in this implementation). Defaults to ''
            sep (str, optional): Separator for nested keys. Defaults to '_'
            
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items: list = []
        for k, v in d.items():
            new_key: str = k  # Use the current key directly, without adding the parent key
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        flattened: Dict[str, Any] = {}
        for key, value in items:
            if key in flattened:
                # NCFM 将嵌套 YAML 展平成 argparse 属性；同名键若静默覆盖，
                # 最终运行值会依赖 YAML 顺序，无法审计，因此直接失败。
                raise ValueError(
                    f"配置存在同名键冲突: '{key}'，请合并字段或改名后再运行。"
                )
            flattened[key] = value
        return flattened

    def add_args_from_yaml(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Add contents of YAML configuration file to args object.
        
        Args:
            args (argparse.Namespace): Argument namespace to update
            
        Returns:
            argparse.Namespace: Updated argument namespace
        """
        # Read the YAML configuration file
        # 配置文件统一使用 UTF-8；Windows 默认 GBK 会把中文注释误判为
        # 编码错误，导致 NCFM 在真正启动前就失败。
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        # Flatten the configuration dictionary
        flat_config: Dict[str, Any] = self.flatten_dict(config)

        # Convert value types (handle floating point numbers and booleans)
        for key, value in flat_config.items():
            # Convert to float if possible
            if isinstance(value, str):
                if value.lower() in ['true', 'false']:
                    flat_config[key] = value.lower() == 'true'
                elif 'e' in value or '.' in value:
                    try:
                        flat_config[key] = float(value)
                    except ValueError:
                        pass

        # Add the flattened configuration items to args
        for key, value in flat_config.items():
            setattr(args, key, value)

        return args
