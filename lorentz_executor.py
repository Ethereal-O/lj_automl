"""
compute_engine/lorentz_executor.py

Handles Lorentz program execution and result parsing.
"""

import glob
import logging
import os
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from factor_factory.data_models.metrics import EvaluationMetrics
from factor_factory.utils.expression_utils import generate_expression_id

class LorentzExecutor:
    """Executes Lorentz program and monitors execution."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def execute(self, cfg_file_path: str) -> Tuple[bool, str]:
        """
        Execute Lorentz program.

        Args:
            cfg_file_path: Path to the .cfg configuration file

        Returns:
            Tuple of (success, error_message)
        """
        try:
            cmd = [self.config.lorentz_executable, cfg_file_path]
            self.logger.info(f"Executing Lorentz: {' '.join(cmd)}")

            # Execute Lorentz with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600, # 1 hour timeout
                check=False
            )

            if result.returncode == 0:
                self.logger.info("Lorentz execution completed successfully")
                return True, ""
            else:
                error_msg = f"Lorentz failed with return code {result.returncode}: {result.stderr}"
                self.logger.error(error_msg)
                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = "Lorentz execution timed out"
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to execute Lorentz: {e}"
            self.logger.error(error_msg)
            return False, error_msg

class LorentzResultParser:
    """Parses Lorentz output files and calculates metrics."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def parse_results(self, optimized_plan) -> Dict[str, Dict]:
        """
        Parse Lorentz output files and calculate metrics.

        Args:
            optimized_plan: Optimized plan from expression optimizer

        Returns:
            Dictionary mapping factor names to metrics
        """
        factors = optimized_plan.get("factors", {})

        # 获取需要输出的因子名称
        output_factor_names = [
            factor_name for factor_name, factor_info in factors.items()
            if factor_info.get("output", True)
        ]

        if not output_factor_names:
            return {}
        try:
            # 新方法: 批量处理所有因子
            return self._calculate_batch_metrics(output_factor_names)
        except Exception as e:
            self.logger.error(f"Batch metrics calculation failed: {e}")
            # 降级到单个因子处理 (保持向后兼容)
            return self._calculate_individual_metrics(output_factor_names)

    def _calculate_batch_metrics(self, factor_names: List[str]) -> Dict[str, Dict]:
        """ 
        Calculate metrics for all factors using batch processing.
        This is the correct approach for Lorentz multi-factor output files.
        """ 
        results = {}

        # 查找所有输出文件
        output_files = self.find_all_output_files()

        if not output_files:
            raise ValueError("No output files found")

        # 为每个因子初始化结果
        for factor_name in factor_names:
            results[factor_name] = {
                "success": False,
                "metrics": {},
                "output_path": "",
                "error": "No data found"
            }

        # 处理每个输出文件 (每个文件包含多个因子)
        section_metrics = {}
        # {factor_name: [section_metrics]} 

        for file_path in output_files:
            try:
                file_metrics = self._calculate_multi_factor_section_metrics(file_path, factor_names)

                # 将文件级别的metrics分配给各个因子
                for factor_name, metrics in file_metrics.items():
                    if factor_name not in section_metrics:
                        section_metrics[factor_name] = []

                    section_metrics[factor_name].append(metrics)
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {e}")
                continue

        # 聚合每个因子的metrics
        for factor_name in factor_names:
            if factor_name in section_metrics and section_metrics[factor_name]:
                try:
                    aggregate_metrics = self._aggregate_section_metrics(section_metrics[factor_name])
                    results[factor_name] = {
                        "success": True,
                        "metrics": aggregate_metrics,
                        "output_path": output_files[0] if output_files else "",
                        "error": None
                    }
                except Exception as e:
                    results[factor_name] = {
                        "success": False,
                        "metrics": {},
                        "output_path": "",
                        "error": f"Aggregation failed: {e}"
                    }
        return results

    def _calculate_individual_metrics(self, factor_names: List[str]) -> Dict[str, Dict]:
        """
        Calculate metrics for each factor individually (fallback method).
        """
        results = {}

        for factor_name in factor_names:
            try:
                metrics = self._calculate_factor_metrics(factor_name)
                results[factor_name] = {
                    "success": True,
                    "metrics": metrics,
                    "output_path": self._get_factor_output_path(factor_name),
                    "error": None
                }
            except Exception as e:
                self.logger.error(f"Failed to calculate metrics for {factor_name}: {e}")
                results[factor_name] = {
                    "success": False,
                    "metrics": {},
                    "output_path": "",
                    "error": str(e)
                }

        return results

    def _calculate_factor_metrics(self, factor_name: str) -> Dict[str, float]:
        """Calculate metrics for a single factor."""
        # Find all CSV files for this factor
        factor_files = self._find_factor_files(factor_name)

        if not factor_files:
            raise ValueError(f"No output files found for factor {factor_name}")

        # Calculate metrics for each cross-section
        section_metrics = []

        for file_path in factor_files:
            try:
                section_metric = self._calculate_section_metrics(file_path)
                if section_metric is not None:
                    section_metrics.append(section_metric)
            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")
                continue

        if not section_metrics:
            raise ValueError(f"No valid metrics calculated for factor {factor_name}")

        # Aggregate metrics across all sections
        return self._aggregate_section_metrics(section_metrics)

    def _find_factor_files(self, factor_name: str) -> List[str]:
        """Find all output files for a factor."""
        # Pattern: /output_root/AutoML/YYYY/YYYYMMDD/eHHMMSS.csv
        pattern = os.path.join(
            self.config.output_factor_root,
            "AutoML",  # Module name
            "*", "*", "*.csv"
        )

        all_files = glob.glob(pattern, recursive=True)
        # filter files that contain this factor
        factor_files = []
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path, nrows=1)  # Just read header
                if factor_name in df.columns:
                    factor_files.append(file_path)
            except Exception:
                continue
        return sorted(factor_files)

    def find_all_output_files(self) -> List[str]:
        """Find all Lorentz output files."""
        # Pattern: /output_root/AutoML/YYYY/YYYYMMDD/eHHMMSS.csv
        pattern = os.path.join(
            self.config.output_factor_root,
            "AutoML",  # Module name
            "*", "*", "*.csv"
        )
        all_files = glob.glob(pattern, recursive=True)
        return sorted(all_files)

    def _calculate_multi_factor_section_metrics(self, file_path: str, factor_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for multiple factors from a single file using efficient batch processing.
        This is the correct approach for Lorentz output files.
        """
        try:
            # 读取包含多个因子的CSV文件
            df = pd.read_csv(file_path)
            if len(df) < 10:  # 需要最少样本数
                return {}
            # 检查是否有forward_return列（实际实现中需要加载真实的return数据）
            if 'forward_return' not in df.columns:
                # 这里应该加载真实的return数据
                # 暂时跳过没有return数据的文件
                return {}

            # 获取存在的因子列
            existing_factors = [name for name in factor_names if name in df.columns]

            if not existing_factors:
                return {}
            
            # 准备因子数据和return数据
            factor_data = df[existing_factors].dropna()
            return_data = df['forward_return'].dropna()

            # 对齐数据（取交集）
            common_index = factor_data.index.intersection(return_data.index)

            if len(common_index) < 10:
                return {}
            aligned_factors = factor_data.loc[common_index]
            aligned_returns = return_data.loc[common_index]

            # 使用corrwith批量计算所有因子的IC（高效方法）
            ic_series = aligned_factors.corrwith(aligned_returns)

            # 构建结果
            results = {}
            for factor_name in existing_factors:
                ic_value = ic_series.get(factor_name)
                if not pd.isna(ic_value):
                    results[factor_name] = {
                        'ic': ic_value,
                        'sample_count': len(common_index)
                    }
            return results

        except Exception as e:
            self.logger.warning(f"Error processing multi-factor file {file_path}: {e}")
            return {}
    
    def _calculate_section_metrics(self, file_path: str) -> Optional[Dict[str, float]]:
        """Calculate metrics for a single cross-section file."""
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            if len(df) < 10: # Need minimum samples
                return None

            # Assume the CSV has columns: symbol, factor_value, forward_return
            # In real implementation, you'd need to join with return data

            # For now, simulate forward returns (this should be replaced with real data)
            if 'forward_return' not in df.columns:
                # This is a placeholder - in real implementation, you'd load actual return data
                df['forward_return'] = pd.Series(index=df.index, dtype=float).fillna(0)
                return None # Skip if no return data

            # Get factor values (assuming single factor per file for simplicity)
            factor_cols = [col for col in df.columns if col not in ['symbol', 'forward_return']]
            if not factor_cols:
                return None
            
            factor_col = factor_cols[0] # Take first factor column
            factor_values = df[factor_col].dropna()
            forward_returns = df['forward_return'].dropna()

            # Align data
            common_index = factor_values.index.intersection(forward_returns.index)
            if len(common_index) < 10:
                return None

            factor_values = factor_values[common_index]
            forward_returns = forward_returns[common_index]

            # Calculate IC and Rank IC
            ic = factor_values.corr(forward_returns)
            rank_ic = factor_values.rank().corr(forward_returns.rank())

            return {
                'ic': ic if not pd.isna(ic) else 0,
                # 'rank_ic': rank_ic if not pd.isna(rank_ic) else 0,
                # 'sample_count': len(common_index)
            }

        except Exception as e:
            self.logger.warning(f"Error processing {file_path}: {e}")
            return None
    
    def _aggregate_section_metrics(self, section_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all sections."""
        if not section_metrics:
            return {}

        # Convert to DataFrame for easy aggregation
        df = pd.DataFrame(section_metrics)

        # Calculate aggregated metrics
        aggregated = {
            'ic': df['ic'].mean(), # 使用标准的'ic'键名
            'ic_std': df['ic'].std() if len(df) > 1 else 0,
            'ic_ir': df['ic'].mean() / df['ic'].std() if len(df) > 1 and df['ic'].std() > 0 else 0,
            'win_rate': (df['ic'] > 0).mean(),
            'total_sections': len(section_metrics),
        }
        
        # 添加样本数统计（如果有的话）
        if 'sample_count' in df.columns:
            aggregated['total_samples'] = df['sample_count'].sum()

        # 计算Sharpelt率（简化版本）
        if aggregated['ic_std'] > 0:
            aggregated['sharpe_ratio'] = aggregated['ic'] / aggregated['ic_std']
        else:
            aggregated['sharpe_ratio'] = 0

        return aggregated

    def _get_factor_output_path(self, factor_name: str) -> str:
        """Get the output path pattern for a factor."""
        return os.path.join(self.config.output_factor_root, "AutoML", "**", "*", "*", f"*[{factor_name}].csv")