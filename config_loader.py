"""
AlphaQCM 配置加载器
从 alphaqcm_config.yaml 文件加载所有超参数
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class AlphaQCMConfig:
    """AlphaQCM 配置管理器"""

    def __init__(self, config_path: str = "alphaqcm_config.yaml"):
        self.config_path = config_path
        self._config = {}
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # 验证配置完整性
        self._validate_config()

    def _validate_config(self):
        """验证配置文件的完整性"""
        required_sections = [
            'training', 'alpha_pool', 'environment', 'data', 'lorentz', 'paths'
        ]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"配置文件缺少必需的配置节: {section}")

    def get(self, key_path: str, default=None):
        """
        获取配置值
        支持点分隔的路径，如 'training.model' 或 'alpha_pool.capacity'

        Args:
            key_path: 配置路径，如 'training.model'
            default: 默认值

        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            if default is not None:
                return default
            raise KeyError(f"配置项不存在: {key_path}")

    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self._config['training']

    def get_alpha_pool_config(self) -> Dict[str, Any]:
        """获取AlphaPool配置"""
        return self._config['alpha_pool']

    def get_environment_config(self) -> Dict[str, Any]:
        """获取环境配置"""
        return self._config['environment']

    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self._config['data']

    def get_lorentz_config(self) -> Dict[str, Any]:
        """获取Lorentz配置"""
        return self._config['lorentz']

    def get_paths_config(self) -> Dict[str, Any]:
        """获取路径配置"""
        return self._config['paths']

    def get_operators_config(self) -> Dict[str, Any]:
        """获取算子配置"""
        return self._config.get('operators', {})

    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return self._config.get('cache', {})

    def get_debug_config(self) -> Dict[str, Any]:
        """获取调试配置"""
        return self._config.get('debug', {})

    def reload(self):
        """重新加载配置文件"""
        self._load_config()

    def __getitem__(self, key):
        """支持字典风格的访问"""
        return self._config[key]

    def __contains__(self, key):
        """检查配置节是否存在"""
        return key in self._config


# 全局配置实例
_config_instance = None

def get_config(config_path: str = "alphaqcm_config.yaml") -> AlphaQCMConfig:
    """
    获取全局配置实例
    使用单例模式确保配置只加载一次
    """
    global _config_instance
    if _config_instance is None or _config_instance.config_path != config_path:
        _config_instance = AlphaQCMConfig(config_path)
    return _config_instance


def load_config_for_train_qcm(args):
    """
    为train_qcm.py加载配置并更新args
    优先级: 命令行参数 > 配置文件 > 默认值
    """
    config = get_config()

    # 训练配置
    training_config = config.get_training_config()

    # 更新args（只在未指定命令行参数时使用配置文件值）
    if not hasattr(args, 'model') or args.model is None:
        args.model = training_config.get('model', 'iqn')
    if not hasattr(args, 'seed') or args.seed is None:
        args.seed = training_config.get('seed', 0)
    if not hasattr(args, 'pool') or args.pool is None:
        args.pool = training_config.get('pool_capacity', 30)
    if not hasattr(args, 'std_lam') or args.std_lam is None:
        args.std_lam = training_config.get('std_lam', 1.0)

    return config


def load_config_for_external_compute():
    """
    为external_compute.py加载配置
    """
    config = get_config()
    data_config = config.get_data_config()
    lorentz_config = config.get_lorentz_config()

    # 更新环境变量
    os.environ['START_DATE'] = data_config.get('start_date', '20200101')
    os.environ['END_DATE'] = data_config.get('end_date', '20241231')
    os.environ['FACTOR_DATA_ROOT_DIR'] = data_config.get('returns_data_root', './returns')
    os.environ['FREQUENCY_CONFIG'] = data_config.get('frequency_config', '1dper1d')
    os.environ['LORENTZ_EXECUTABLE'] = lorentz_config.get('executable_path', '/dfs/dataset/365-1734663142170/data/Lorentz_History-Insider')
    os.environ['THREAD_NUM'] = str(lorentz_config.get('thread_num', 8))

    return config


# 便捷函数
def get_training_config():
    """获取训练配置"""
    return get_config().get_training_config()

def get_alpha_pool_config():
    """获取AlphaPool配置"""
    return get_config().get_alpha_pool_config()

def get_environment_config():
    """获取环境配置"""
    return get_config().get_environment_config()

def get_data_config():
    """获取数据配置"""
    return get_config().get_data_config()

def get_lorentz_config():
    """获取Lorentz配置"""
    return get_config().get_lorentz_config()

def get_paths_config():
    """获取路径配置"""
    return get_config().get_paths_config()
