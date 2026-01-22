#!/usr/bin/env python3
"""
External Factor Computation Script
使用Lorentz程序计算alpha因子值并输出CSV格式结果
"""

import sys
import os
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LorentzConfig:
    """Lorentz配置管理"""

    def __init__(self):
        # 尝试加载AlphaQCM配置
        try:
            from config_loader import load_config_for_external_compute
            alphaqcm_config = load_config_for_external_compute()
            data_config = alphaqcm_config.get_data_config()
            lorentz_config = alphaqcm_config.get_lorentz_config()
            paths_config = alphaqcm_config.get_paths_config()

            # 使用配置文件参数
            self.lorentz_executable = lorentz_config.get('executable_path', '/dfs/dataset/365-1734663142170/data/Lorentz_History-Insider')
            self.thread_num = lorentz_config.get('thread_num', 8)
            self.start_date = data_config.get('start_date', '20200101')
            self.end_date = data_config.get('end_date', '20241231')
            self.frequency_config = data_config.get('frequency_config', '1dper1d')

            # 使用配置文件中的路径设置
            self.output_factor_root_dir = paths_config.get('factors_output_dir', '/dfs/data/Factors')
            self.output_abnormal_root_dir = paths_config.get('abnormal_stats_dir', '/dfs/data/AbnormalStats')

        except Exception:
            # 如果配置加载失败，使用环境变量或默认值
            self.lorentz_executable = os.getenv('LORENTZ_EXECUTABLE', '/dfs/dataset/365-1734663142170/data/Lorentz_History-Insider')
            self.thread_num = int(os.getenv('THREAD_NUM', '8'))
            self.start_date = os.getenv('START_DATE', '20200101')
            self.end_date = os.getenv('END_DATE', '20241231')
            self.frequency_config = os.getenv('FREQUENCY_CONFIG', '1dper1d')
            self.output_factor_root_dir = os.getenv('OUTPUT_FACTOR_ROOT_DIR', '/dfs/data/Factors')
            self.output_abnormal_root_dir = os.getenv('OUTPUT_ABNORMAL_ROOT_DIR', '/dfs/data/AbnormalStats')

        # 基于频率配置动态生成路径
        base_data_path = '/dfs/dataset/365-1734663142170/data'

        # 动态生成路径（如果没有在配置中指定）
        self.interval_json = os.getenv('INTERVAL_JSON',
            f'{base_data_path}/LorentzConfigTemplate/{self.frequency_config}/interval_{self.frequency_config}.json')
        self.data_root_dir = os.getenv('DATA_ROOT_DIR',
            f'{base_data_path}/BasicFieldsDump-Latest-Release/{self.frequency_config}')
        self.daily_data_dir = os.getenv('DAILY_DATA_DIR',
            f'{base_data_path}/BasicFieldsDump-Latest-Release/{self.frequency_config}')

        # 确保输出目录包含频率配置
        if not self.output_factor_root_dir.endswith(self.frequency_config):
            self.output_factor_root_dir = os.path.join(self.output_factor_root_dir, self.frequency_config)
        if not self.output_abnormal_root_dir.endswith(self.frequency_config):
            self.output_abnormal_root_dir = os.path.join(self.output_abnormal_root_dir, self.frequency_config)


class LorentzExecutor:
    """Lorentz程序执行器"""

    def __init__(self, config: LorentzConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def execute_for_date(self, date_str: str, factor_json_path: str, output_names_path: str,
                        output_module_name: str, load_prev_days: int = 1) -> Tuple[bool, str]:
        """
        为指定日期执行Lorentz计算

        Args:
            date_str: 计算日期 (YYYYMMDD格式)
            factor_json_path: 因子JSON配置文件路径
            output_names_path: 输出因子名称文件路径
            output_module_name: 输出模块名称
            load_prev_days: 加载前N天的参数

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # 生成临时配置文件 - 保存到debug目录以便检查
            cfg_content = self._generate_cfg_content(
                date_str, factor_json_path, output_names_path, output_module_name, load_prev_days
            )

            # 创建debug目录
            debug_dir = os.path.join(os.getcwd(), 'lorentz_debug')
            os.makedirs(debug_dir, exist_ok=True)

            cfg_file_path = os.path.join(debug_dir, f'lorentz_config_{date_str}.cfg')
            with open(cfg_file_path, 'w') as cfg_file:
                cfg_file.write(cfg_content)

            try:
                # 执行Lorentz程序
                cmd = [self.config.lorentz_executable, cfg_file_path]

                # 简化的开始标记
                print(f"\n=== LORENTZ START ({date_str}) ===", file=sys.stderr)

                # 使用os.system确保输出可见
                import os
                cmd_str = ' '.join(cmd)
                return_code = os.system(cmd_str)

                # 简化的结束标记
                print(f"=== LORENTZ END (code: {return_code}) ===\n", file=sys.stderr)

                # 模拟subprocess.CompletedProcess
                class MockCompletedProcess:
                    def __init__(self, returncode):
                        self.returncode = returncode
                        self.stdout = ""
                        self.stderr = ""

                result = MockCompletedProcess(return_code)

                # 为了向后兼容，设置空的stdout/stderr
                result.stdout = ""
                result.stderr = ""

                if result.returncode == 0:
                    self.logger.info(f"Lorentz execution completed successfully for {date_str}")
                    return True, ""
                else:
                    error_msg = f"Lorentz failed with return code {result.returncode}: {result.stderr}"
                    self.logger.error(error_msg)
                    return False, error_msg

            finally:
                # 清理临时文件
                try:
                    os.unlink(cfg_file_path)
                except:
                    pass

        except subprocess.TimeoutExpired:
            error_msg = f"Lorentz execution timed out for {date_str}"
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to execute Lorentz for {date_str}: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    def _generate_cfg_content(self, date_str: str, factor_json_path: str,
                            output_names_path: str, output_module_name: str, load_prev_days: int = 1) -> str:
        """生成Lorentz配置文件内容"""
        cfg_lines = [
            f"DATE={date_str}",
            f"INTERVAL_JSON={self.config.interval_json}",
            "",
            "[BasicFields]",
            f"DATA_ROOT_DIR={self.config.data_root_dir}",
            f"LOAD_PREV_DAYS={load_prev_days}",
            f"THREAD_NUM={self.config.thread_num}",
            f"AUTO_PROD_CO_DEPENDENCY=TRUE",
            f"DAILY_DATA_DIR={self.config.daily_data_dir}",
            "",
            "[ComputeGraph]",
            f"THREAD_NUM={self.config.thread_num}",
            f"FACTOR_JSON={factor_json_path}",
            f"OUTPUT_MODULE_NAME={output_module_name}",
            f"OUTPUTS_CONFIG_FILES={output_names_path}",
            f"EMABLE_OUTPUT_CSV=TRUE",
            f"CSV_FLOAT_PRECISION=6",
            f"OUTPUT_FACTOR_ROOT_DIR={self.config.output_factor_root_dir}",
            f"OUTPUT_ABNORMAL_ROOT_DIR={self.config.output_abnormal_root_dir}",
        ]
        return "\n".join(cfg_lines)


class LorentzResultParser:
    """Lorentz结果解析器"""

    def __init__(self, config: LorentzConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _hhmmss_to_minutecode(self, hhmmss: str) -> int:
        """
        将HHMMSS格式转换为minuteCode (0-240)

        Args:
            hhmmss: 时间戳字符串，如 "093000"

        Returns:
            minuteCode: 0-240的分钟编码
        """
        try:
            # 解析小时和分钟
            hour = int(hhmmss[:2])
            minute = int(hhmmss[2:4])

            # 计算从9:30开始的分钟数
            # 9:30 = 0分钟
            # 11:30 = 120分钟 (中午休息)
            # 13:00 = 121分钟 (下午开盘)
            # 15:00 = 240分钟 (收盘)

            if hour < 12:  # 上午
                total_minutes = (hour - 9) * 60 + minute - 30
            else:  # 下午
                # 下午1:00开始，减去中午休息时间
                total_minutes = 120 + (hour - 13) * 60 + minute

            return int(total_minutes)

        except (ValueError, IndexError):
            self.logger.error(f"Invalid HHMMSS format: {hhmmss}")
            return -1

    def parse_factor_output(self, date_str: str, factor_name: str) -> Optional[pd.DataFrame]:
        """
        解析指定日期和因子的所有输出文件

        Args:
            date_str: 日期字符串 (YYYYMMDD)
            factor_name: 因子名称

        Returns:
            包含symbol, minuteCode, factor_value的DataFrame，如果解析失败则返回None
        """
        try:
            # 构建输出文件路径
            # 格式: /output_root/AutoML/YYYY/YYYYMMDD/eHHMMSS.csv
            year = date_str[:4]
            output_pattern = os.path.join(
                self.config.output_factor_root_dir,
                "AutoML",
                year,
                date_str,
                "e*.csv"  # 匹配所有e开头的时间戳文件
            )

            import glob
            output_files = glob.glob(output_pattern)

            if not output_files:
                self.logger.warning(f"No output files found for {date_str}")
                return None

            # 读取所有时间点的文件
            all_results = []

            for file_path in sorted(output_files):  # 按时间排序
                try:
                    # 从文件名提取时间戳
                    filename = os.path.basename(file_path)
                    if not filename.startswith('e') or not filename.endswith('.csv'):
                        continue

                    hhmmss = filename[1:-4]  # 去掉'e'和'.csv'
                    minute_code = self._hhmmss_to_minutecode(hhmmss)

                    if minute_code < 0:
                        continue

                    # 读取CSV文件
                    df = pd.read_csv(file_path)

                    # 检查是否包含所需的因子列
                    if factor_name not in df.columns:
                        self.logger.warning(f"Factor {factor_name} not found in {file_path}")
                        continue

                    # 提取数据
                    temp_df = df[['symbol', factor_name]].copy()
                    temp_df['date'] = pd.to_datetime(date_str, format='%Y%m%d')
                    temp_df['minuteCode'] = minute_code
                    temp_df = temp_df.rename(columns={factor_name: 'factor_value'})

                    all_results.append(temp_df)

                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")
                    continue

            if not all_results:
                self.logger.error(f"No valid factor data found for {date_str}")
                return None

            # 合并所有时间点的数据
            combined_df = pd.concat(all_results, ignore_index=True)

            # 记录数据统计信息
            actual_minute_codes = sorted(combined_df['minuteCode'].unique())
            n_time_points = len(actual_minute_codes)

            self.logger.info(f"Parsed {len(combined_df)} factor values for {date_str} across {n_time_points} time points: {actual_minute_codes[:5]}...{actual_minute_codes[-5:]}")

            # 对于per1以外的配置，检查时间点间隔是否合理
            if n_time_points > 1:
                intervals = np.diff(actual_minute_codes)
                avg_interval = np.mean(intervals)
                self.logger.info(f"Average time interval: {avg_interval:.1f} minutes")

            return combined_df

        except Exception as e:
            self.logger.error(f"Error parsing output for {date_str}, factor {factor_name}: {e}")
            return None

    def parse_batch_factor_output(self, date_str: str, factor_names: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
        """
        解析批量因子输出，返回所有因子的结果

        Args:
            date_str: 日期字符串 (YYYYMMDD)
            factor_names: 因子名称列表

        Returns:
            字典：因子名称 -> 包含数据的DataFrame
        """
        try:
            # 构建输出文件路径
            year = date_str[:4]
            output_pattern = os.path.join(
                self.config.output_factor_root_dir,
                "AutoML",
                year,
                date_str,
                "e*.csv"
            )

            import glob
            output_files = glob.glob(output_pattern)

            if not output_files:
                self.logger.warning(f"No output files found for {date_str}")
                return None

            # 为每个因子收集数据
            factor_results = {name: [] for name in factor_names}

            # 读取所有时间点的文件
            for file_path in sorted(output_files):
                try:
                    # 从文件名提取时间戳
                    filename = os.path.basename(file_path)
                    if not filename.startswith('e') or not filename.endswith('.csv'):
                        continue

                    hhmmss = filename[1:-4]
                    minute_code = self._hhmmss_to_minutecode(hhmmss)

                    if minute_code < 0:
                        continue

                    # 读取CSV文件
                    df = pd.read_csv(file_path)

                    # 为每个因子提取数据
                    for factor_name in factor_names:
                        if factor_name in df.columns:
                            temp_df = df[['symbol', factor_name]].copy()
                            temp_df['date'] = pd.to_datetime(date_str, format='%Y%m%d')
                            temp_df['minuteCode'] = minute_code
                            temp_df = temp_df.rename(columns={factor_name: 'factor_value'})

                            factor_results[factor_name].append(temp_df)

                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")
                    continue

            # 合并每个因子的数据
            result = {}
            for factor_name, dfs in factor_results.items():
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    result[factor_name] = combined_df
                else:
                    self.logger.warning(f"No data found for factor {factor_name}")

            if result:
                # 记录统计信息
                total_records = sum(len(df) for df in result.values())
                self.logger.info(f"Parsed {total_records} records for {len(result)} factors on {date_str}")
                return result
            else:
                self.logger.error(f"No valid factor data found for {date_str}")
                return None

        except Exception as e:
            self.logger.error(f"Error parsing batch output for {date_str}: {e}")
            return None


def convert_compact_operators_to_lorentz(expr_str: str) -> str:
    """
    将写死参数的算子转换回Lorentz能理解的原始格式

    Args:
        expr_str: 包含写死参数算子的表达式字符串

    Returns:
        转换后的表达式字符串，算子恢复为原始格式
    """
    import re

    def replace_compact_operator(match):
        """替换单个写死参数算子"""
        compact_op = match.group(0)

        # 首先处理Ts时序算子
        pattern_ts = r'^Ts(\w+)(\d+)([FT])$'
        match_ts = re.match(pattern_ts, compact_op)

        if match_ts:
            op_name = match_ts.group(1)  # 基础操作名
            window = int(match_ts.group(2))  # 窗口大小
            bias_flag = match_ts.group(3)  # F或T

            # 转换bias
            bias = False if bias_flag == 'F' else True

            # 构建原始格式：OpName(x, window, bias)
            return f"Ts{op_name}(x, {window}, {str(bias).lower()})"

        # 处理CsWinsorize算子
        pattern_winsorize = r'^CsWinsorize(\d+)$'
        match_winsorize = re.match(pattern_winsorize, compact_op)

        if match_winsorize:
            std_ratio = int(match_winsorize.group(1)) / 10  # 05->0.5, 10->1.0, etc.
            return f"CsWinsorize(x, {std_ratio}, group)"

        # 处理CsRangeMask算子
        pattern_range = r'^CsRangeMask([LUD])([KR])(\d+)$'
        match_range = re.match(pattern_range, compact_op)

        if match_range:
            border = match_range.group(1)  # L/U/D
            op_type = match_range.group(2)  # K/R
            pct = int(match_range.group(3))  # 01/05/10/25

            if border == 'L':  # Lower边
                if op_type == 'K':  # 要极值
                    lower_pct, upper_pct = 0, pct
                else:  # 去极值
                    lower_pct, upper_pct = pct, 100
            elif border == 'U':  # Upper边
                if op_type == 'K':  # 要极值
                    lower_pct, upper_pct = 100 - pct, 100
                else:  # 去极值
                    lower_pct, upper_pct = 0, 100 - pct
            else:  # 双边 D
                if op_type == 'K':  # 要极值
                    lower_pct, upper_pct = pct, 100 - pct
                else:  # 去极值
                    # 对于双边去极值，我们使用单个范围表示（需要特殊处理）
                    lower_pct, upper_pct = 0, pct

            return f"CsRangeMask(x, {lower_pct}, {upper_pct}, substitute, mask, group)"

        return compact_op

    # 使用正则表达式替换所有写死参数算子
    # 匹配各种写死参数算子
    patterns = [
        r'\bTs\w+\d+[FT]\b',        # Ts开头的时序算子
        r'\bCsWinsorize\d+\b',      # CsWinsorize算子
        r'\bCsRangeMask\w+\d+\b',   # CsRangeMask算子
    ]

    result = expr_str
    for pattern in patterns:
        result = re.sub(pattern, replace_compact_operator, result)

    return result


def parse_alpha_expression(expr_str: str) -> Dict[str, str]:
    """
    解析alpha表达式并生成Lorentz配置
    将写死参数的算子转换回Lorentz能理解的原始格式

    Args:
        expr_str: Alpha表达式字符串

    Returns:
        包含因子名称和表达式的字典
    """
    # 将写死参数的算子转换回原始格式
    converted_expr_str = convert_compact_operators_to_lorentz(expr_str)

    # 生成因子名称 (使用表达式哈希作为唯一标识)
    import hashlib
    factor_name = f"Factor_{hashlib.md5(converted_expr_str.encode()).hexdigest()[:8]}"

    return {
        "factor_name": factor_name,
        "expression": converted_expr_str
    }


def convert_field_references(expr_str: str) -> str:
    """
    转换表达式中的字段引用，添加@前缀

    Args:
        expr_str: 原始表达式字符串

    Returns:
        转换后的表达式字符串
    """
    # 匹配字段引用模式，如 $Slice.LastPrice, $Preload.Volume 等
    import re

    def replace_field(match):
        field_ref = match.group(0)
        # 去掉开头的$，加上@
        return '@' + field_ref[1:]

    # 匹配 $后跟字母的字段引用
    pattern = r'\$[A-Za-z][A-Za-z0-9_.]*'
    return re.sub(pattern, replace_field, expr_str)


def analyze_lookback_requirements(expr_str: str) -> Dict[str, int]:
    """
    分析表达式中的Ts算子，计算回看时间要求

    Args:
        expr_str: 表达式字符串

    Returns:
        包含rolling_prev_days和rolling_prev_intervals的字典
    """
    import re

    rolling_prev_days = 0
    rolling_prev_intervals = 0

    # 使用栈来解析嵌套的函数调用
    def parse_function_calls(text: str):
        """解析所有函数调用，返回(func_name, args_str)列表"""
        calls = []
        i = 0
        while i < len(text):
            # 查找函数名
            if text[i].isalpha():
                # 找到函数名开始
                start = i
                while i < len(text) and (text[i].isalnum() or text[i] == '_'):
                    i += 1
                func_name = text[start:i]

                # 查找对应的左括号
                while i < len(text) and text[i] != '(':
                    i += 1

                if i < len(text) and text[i] == '(':
                    # 找到左括号，开始解析参数
                    paren_count = 1
                    arg_start = i + 1
                    i += 1

                    while i < len(text) and paren_count > 0:
                        if text[i] == '(':
                            paren_count += 1
                        elif text[i] == ')':
                            paren_count -= 1
                        i += 1

                    if paren_count == 0:  # 找到匹配的右括号
                        args_str = text[arg_start:i-1]
                        calls.append((func_name, args_str))

                        # 递归解析参数中的嵌套调用
                        calls.extend(parse_function_calls(args_str))
            else:
                i += 1

        return calls

    # 解析所有函数调用
    all_calls = parse_function_calls(expr_str)

    # 只处理Ts开头的算子
    ts_calls = [(name, args) for name, args in all_calls if name.startswith('Ts')]

    for func_name, args_str in ts_calls:
        # 解析参数
        args = [arg.strip() for arg in args_str.split(',') if arg.strip()]

        # 需要至少3个参数
        if len(args) < 3:
            continue

        try:
            # 倒数第二个参数：时间窗口
            time_window = int(args[-2])

            # 最后一个参数：是否跨日
            last_arg = args[-1].lower().strip()
            is_cross_day = last_arg in ['true', '1', 'yes']

            if is_cross_day:
                # 跨日：使用rolling_prev_days
                rolling_prev_days = max(rolling_prev_days, time_window)
            else:
                # 同日：使用rolling_prev_intervals
                rolling_prev_intervals = max(rolling_prev_intervals, time_window)

        except (ValueError, IndexError):
            # 参数解析失败，跳过
            continue

    # 构建结果：rolling_prev_days优先级高于rolling_prev_intervals
    result = {}
    if rolling_prev_days > 0:
        result['rolling_prev_days'] = rolling_prev_days
    elif rolling_prev_intervals > 0:
        result['rolling_prev_intervals'] = rolling_prev_intervals

    return result


def parse_expression_with_intermediates(expr_str: str) -> Dict[str, any]:
    """
    解析表达式，支持中间变量提取
    严格按照Lorentz规则：Cs算子只能在cross_section中，且参数只能是简单引用

    Args:
        expr_str: 原始表达式字符串

    Returns:
        包含slice_expressions, cross_section_expressions, final_expression的字典
    """
    import re

    # 用于生成唯一变量名的计数器
    var_counter = 0

    def get_next_var_name():
        nonlocal var_counter
        var_counter += 1
        return f"var_{var_counter}"

    # 存储中间变量定义
    slice_intermediates = []  # slice中的中间变量（无Cs算子）
    cross_section_intermediates = []  # cross_section中的中间变量（可能包含Cs算子）

    def is_simple_arg(expr: str) -> bool:
        """
        判断Cs算子参数是否为简单参数
        Cs算子参数只能是：变量引用、字段引用、const常量
        """
        expr = expr.strip()

        # 变量引用
        if expr.startswith('@'):
            return True

        # 字段引用
        if expr.replace('.', '').replace('_', '').isalnum() and '.' in expr:
            return True

        # const常量（数字、布尔）
        try:
            float(expr)
            return True
        except:
            pass

        if expr.lower() in ['true', 'false']:
            return True

        return False

    def extract_subexpressions(text: str) -> str:
        """
        提取复杂子表达式，确保Cs算子及其复杂参数都在cross_section中
        """
        result = text

        # 第一步：处理所有非Cs算子，提取复杂参数为slice中间变量
        def process_non_cs_functions(expr: str) -> str:
            """处理非Cs算子，提取复杂参数"""
            res = expr

            # 使用栈解析函数调用
            def parse_function_calls(e: str):
                calls = []
                i = 0
                while i < len(e):
                    if e[i].isalpha():
                        func_start = i
                        while i < len(e) and (e[i].isalnum() or e[i] == '_'):
                            i += 1

                        func_name = e[func_start:i]

                        if func_name.startswith('Cs'):
                            # 跳过Cs算子，在后续处理
                            continue

                        if i < len(e) and e[i] == '(':
                            paren_count = 1
                            args_start = i + 1
                            i += 1

                            while i < len(e) and paren_count > 0:
                                if e[i] == '(':
                                    paren_count += 1
                                elif e[i] == ')':
                                    paren_count -= 1
                                i += 1

                            if paren_count == 0:
                                args_str = e[args_start:i-1]

                                # 解析参数
                                args = []
                                current_arg = ""
                                arg_depth = 0

                                for char in args_str:
                                    if char == '(':
                                        arg_depth += 1
                                        current_arg += char
                                    elif char == ')':
                                        arg_depth -= 1
                                        current_arg += char
                                    elif char == ',' and arg_depth == 0:
                                        if current_arg.strip():
                                            args.append(current_arg.strip())
                                        current_arg = ""
                                    else:
                                        current_arg += char

                                if current_arg.strip():
                                    args.append(current_arg.strip())

                                calls.append((func_name, args, func_start, i))
                    else:
                        i += 1
                return calls

            calls = parse_function_calls(res)
            calls.sort(key=lambda x: x[2], reverse=True)  # 从后往前处理

            for func_name, args, start_pos, end_pos in calls:
                processed_args = []
                for arg in args:
                    if not is_simple_arg(arg):
                        var_name = get_next_var_name()
                        slice_intermediates.append({
                            'name': var_name,
                            'expression': arg,
                            'output': False
                        })
                        processed_args.append(f'@{var_name}')
                    else:
                        processed_args.append(arg)

                # 重建调用
                new_args_str = ','.join(processed_args)
                new_call = f'{func_name}({new_args_str})'
                res = res[:start_pos] + new_call + res[end_pos:]

            return res

        # 第一步：处理非Cs算子
        result = process_non_cs_functions(result)

        # 第二步：处理Cs算子及其复杂参数
        def process_cs_functions(expr: str) -> str:
            """处理Cs算子，将其复杂参数提取为cross_section中间变量"""
            res = expr

            # 查找所有Cs算子
            cs_pattern = r'\b(Cs\w*)\(([^()]*(?:\([^()]*\)[^()]*)*)\)'
            cs_matches = []

            for match in re.finditer(cs_pattern, res):
                func_name = match.group(1)
                args_str = match.group(2)
                start_pos = match.start()
                end_pos = match.end()

                # 解析参数
                args = []
                current_arg = ""
                paren_depth = 0

                for char in args_str:
                    if char == '(':
                        paren_depth += 1
                        current_arg += char
                    elif char == ')':
                        paren_depth -= 1
                        current_arg += char
                    elif char == ',' and paren_depth == 0:
                        if current_arg.strip():
                            args.append(current_arg.strip())
                        current_arg = ""
                    else:
                        current_arg += char

                if current_arg.strip():
                    args.append(current_arg.strip())

                cs_matches.append((func_name, args, start_pos, end_pos))

            # 从后往前处理
            cs_matches.sort(key=lambda x: x[2], reverse=True)

            for func_name, args, start_pos, end_pos in cs_matches:
                processed_args = []
                for arg in args:
                    if not is_simple_arg(arg):
                        # Cs算子的复杂参数，提取为cross_section中间变量
                        var_name = get_next_var_name()
                        cross_section_intermediates.append({
                            'name': var_name,
                            'expression': arg,
                            'output': False
                        })
                        processed_args.append(f'@{var_name}')
                    else:
                        processed_args.append(arg)

                # 重建Cs调用
                new_args_str = ','.join(processed_args)
                new_call = f'{func_name}({new_args_str})'
                res = res[:start_pos] + new_call + res[end_pos:]

            return res

        # 第二步：处理Cs算子
        result = process_cs_functions(result)

        # 第三步：再次处理Cs算子内部的复杂参数（递归处理）
        result = process_cs_functions(result)

        return result

    # 第一遍：提取所有复杂子表达式到slice
    processed_expr = extract_subexpressions(expr_str)

    # 第二遍：检查最终表达式是否需要进一步处理
    # 如果最终表达式包含Cs算子且有复杂参数，需要再次处理
    final_expression = processed_expr
    has_cs_operators = bool(re.search(r'\bCs\w*\(', final_expression))

    return {
        'slice_intermediates': slice_intermediates,
        'cross_section_intermediates': cross_section_intermediates,
        'final_expression': final_expression,
        'has_cs_operators': has_cs_operators
    }


def generate_lorentz_config_files(parsed_expr: Dict[str, str], temp_dir: str) -> Tuple[str, str, str]:
    """
    生成Lorentz需要的配置文件，支持中间变量和cross_section

    Args:
        parsed_expr: 解析后的表达式信息
        temp_dir: 临时目录路径

    Returns:
        Tuple of (factor_json_path, output_names_path, output_module_name)
    """
    factor_name = parsed_expr["factor_name"]
    expression = parsed_expr["expression"]

    # 第一步：转换字段引用（添加@前缀）
    converted_expression = convert_field_references(expression)

    # 第二步：解析表达式，提取中间变量
    parsed_result = parse_expression_with_intermediates(converted_expression)
    slice_intermediates = parsed_result['slice_intermediates']
    cross_section_intermediates = parsed_result['cross_section_intermediates']
    final_expression = parsed_result['final_expression']
    has_cs_operators = parsed_result['has_cs_operators']

    # 第四步：构建配置结构
    default_config = {
        "slice": {
            "trigger": "slice",
            "output": True
        }
    }

    slice_configs = []
    cross_section_configs = []

    # 为每个slice中间变量单独分析回看时间
    for var_config in slice_intermediates:
        expr = var_config['expression']
        lookback_config = analyze_lookback_requirements(expr)
        if lookback_config:
            var_config.update(lookback_config)
        slice_configs.append(var_config)

    # 为每个cross_section中间变量单独分析回看时间
    for var_config in cross_section_intermediates:
        expr = var_config['expression']
        lookback_config = analyze_lookback_requirements(expr)
        if lookback_config:
            var_config.update(lookback_config)
        cross_section_configs.append(var_config)

    # 处理最终表达式
    if has_cs_operators:
        # 有Cs算子：激活cross_section
        default_config["cross_section"] = {
            "trigger": "cross_section",
            "output": True
        }

        # 最终表达式放在cross_section中
        final_config = {
            "name": factor_name,
            "expression": final_expression,
            "trigger": "cross_section",
            "output": True
        }

        # 为最终表达式单独分析回看时间
        final_lookback_config = analyze_lookback_requirements(final_expression)
        if final_lookback_config:
            final_config.update(final_lookback_config)

        cross_section_configs.append(final_config)

    else:
        # 无Cs算子：只放在slice中
        final_config = {
            "name": factor_name,
            "expression": final_expression,
            "trigger": "slice",
            "output": True
        }

        # 为最终表达式单独分析回看时间
        final_lookback_config = analyze_lookback_requirements(final_expression)
        if final_lookback_config:
            final_config.update(final_lookback_config)

        slice_configs.append(final_config)

    # 第五步：构建最终JSON
    factor_json = {
        "default": default_config,
        "slice": slice_configs
    }

    if cross_section_configs:
        factor_json["cross_section"] = cross_section_configs

    factor_json_path = os.path.join(temp_dir, "factor_config.json")
    with open(factor_json_path, 'w', encoding='utf-8') as f:
        json.dump(factor_json, f, indent=2, ensure_ascii=False)

    # 生成输出因子名称文件（只包含output=true的因子）
    output_names = []

    # 收集slice中output=true的因子
    for config in slice_configs:
        if config.get('output', False):
            output_names.append(config['name'])

    # 收集cross_section中output=true的因子
    for config in cross_section_configs:
        if config.get('output', False):
            output_names.append(config['name'])

    output_names_path = os.path.join(temp_dir, "factor_names.txt")
    with open(output_names_path, 'w', encoding='utf-8') as f:
        for name in output_names:
            f.write(f"{name}\n")

    # 输出模块名称
    output_module_name = f"set_{factor_name.split('_')[-1]}"

    return factor_json_path, output_names_path, output_module_name


def compute_factor_values_with_lorentz(parsed_expr: Dict[str, str]) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.Index]:
    """
    使用Lorentz计算因子值

    Args:
        parsed_expr: 解析后的表达式信息

    Returns:
        values: (n_days, n_stocks) 的因子值数组
        dates: 日期索引
        symbols: 股票代码索引
    """
    config = LorentzConfig()
    executor = LorentzExecutor(config)
    parser = LorentzResultParser(config)

    factor_name = parsed_expr["factor_name"]
    expr_str = parsed_expr["expression"]

    print(f"🔧 Lorentz Configuration for: {expr_str}", file=sys.stderr)
    print(f"   Factor name: {factor_name}", file=sys.stderr)

    # 创建debug目录用于保存配置文件
    debug_dir = os.path.join(os.getcwd(), 'lorentz_debug')
    os.makedirs(debug_dir, exist_ok=True)
    print(f"DEBUG: Created lorentz_debug directory at: {debug_dir}", file=sys.stderr)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 生成配置文件
        factor_json_path, output_names_path, output_module_name = generate_lorentz_config_files(
            parsed_expr, temp_dir
        )

        # 复制配置文件到debug目录供查看
        import shutil
        debug_factor_json = os.path.join(debug_dir, 'factor_config.json')
        debug_output_names = os.path.join(debug_dir, 'factor_names.txt')
        shutil.copy2(factor_json_path, debug_factor_json)
        shutil.copy2(output_names_path, debug_output_names)
        print(f"DEBUG: Saved config files to debug directory", file=sys.stderr)

        # ===== 打印配置文件内容 =====
        print(f"\n📋 Lorentz Configuration Files for: {expr_str}", file=sys.stderr)

        # 打印 factor_config.json
        print(f"\n🔧 factor_config.json:", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        try:
            with open(factor_json_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
                print(json_content, file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to read factor_config.json: {e}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        # 打印 factor_names.txt
        print(f"\n📝 factor_names.txt:", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        try:
            with open(output_names_path, 'r', encoding='utf-8') as f:
                txt_content = f.read()
                print(txt_content, file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to read factor_names.txt: {e}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        # 打印 Lorentz 程序信息
        print(f"\n🏭 Lorentz Program Information:", file=sys.stderr)
        print(f"   Executable: {config.lorentz_executable}", file=sys.stderr)
        print(f"   Thread num: {config.thread_num}", file=sys.stderr)
        print(f"   Data root: {config.data_root_dir}", file=sys.stderr)
        print(f"   Output root: {config.output_factor_root_dir}", file=sys.stderr)
        print(f"   Start date: {config.start_date}", file=sys.stderr)
        print(f"   End date: {config.end_date}", file=sys.stderr)
        print(f"   Output module: {output_module_name}", file=sys.stderr)

        # 解析日期范围
        start_date = datetime.strptime(config.start_date, '%Y%m%d')
        end_date = datetime.strptime(config.end_date, '%Y%m%d')

        all_results = []

        # 为每个日期执行计算
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')

            try:
                # 执行Lorentz计算
                success, error_msg = executor.execute_for_date(
                    date_str, factor_json_path, output_names_path, output_module_name
                )

                if success:
                    # 解析结果
                    result_df = parser.parse_factor_output(date_str, factor_name)
                    if result_df is not None:
                        all_results.append(result_df)
                        logger.info(f"Successfully computed factor for {date_str}")
                    else:
                        logger.warning(f"Failed to parse results for {date_str}")
                else:
                    logger.error(f"Failed to compute factor for {date_str}: {error_msg}")

            except Exception as e:
                logger.error(f"Exception during Lorentz computation for {date_str}: {e}")
                import traceback
                traceback.print_exc()

            # 移动到下一天
            current_date += timedelta(days=1)

        if not all_results:
            print(f"ERROR: No factor values were successfully computed for expression {parsed_expr['expression']}", file=sys.stderr)
            raise ValueError("No factor values were successfully computed")

        # 合并所有日期的结果
        combined_df = pd.concat(all_results, ignore_index=True)

        # 转换为numpy数组格式
        # 透视表：行=日期，列=股票代码，值=因子值
        pivot_df = combined_df.pivot(index='date', columns='symbol', values=factor_name)

        # 填充缺失值
        pivot_df = pivot_df.fillna(0.0)

        values = pivot_df.values
        dates = pivot_df.index
        symbols = pivot_df.columns

        logger.info(f"Computed factor values: {values.shape}")

        return values, dates, symbols


def compute_batch_factor_values(parsed_exprs: List[Dict], data_source=None) -> Dict[str, Tuple[np.ndarray, pd.DatetimeIndex, pd.Index]]:
    """
    批量计算多个因子值的主函数

    Args:
        parsed_exprs: 解析后的表达式列表
        data_source: 数据源（可选）

    Returns:
        字典：表达式名称 -> (values, dates, symbols)
    """
    try:
        if len(parsed_exprs) == 1:
            # 单表达式：使用原有逻辑
            values, dates, symbols = compute_factor_values_with_lorentz(parsed_exprs[0])
            return {parsed_exprs[0]["factor_name"]: (values, dates, symbols)}

        # 批量计算多个表达式
        return compute_batch_factor_values_with_lorentz(parsed_exprs)

    except Exception as e:
        # ===== 打印详细的调试信息并终止程序 =====
        print("\n" + "="*80, file=sys.stderr)
        print("🚨 LORENTZ COMPUTATION FAILED - TERMINATING PROGRAM", file=sys.stderr)
        print("="*80, file=sys.stderr)

        # 打印这一批的所有表达式信息
        print(f"\n📋 Batch contained {len(parsed_exprs)} expressions:", file=sys.stderr)
        for i, parsed_expr in enumerate(parsed_exprs, 1):
            expr_str = parsed_expr['expression']
            factor_name = parsed_expr['factor_name']
            print(f"  {i}. Factor: {factor_name}", file=sys.stderr)
            print(f"     Expression: {expr_str}", file=sys.stderr)

        # 尝试生成配置文件并显示内容
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if len(parsed_exprs) == 1:
                    factor_json_path, output_names_path, output_module_name = generate_lorentz_config_files(
                        parsed_exprs[0], temp_dir
                    )
                else:
                    factor_json_path, output_names_path, output_module_name = generate_batch_lorentz_config_files(
                        parsed_exprs, temp_dir
                    )

                print(f"\n📄 Configuration files generated in: {temp_dir}", file=sys.stderr)

                # 显示factor_config.json内容
                print(f"\n📋 factor_config.json content:", file=sys.stderr)
                print("-" * 40, file=sys.stderr)
                with open(factor_json_path, 'r', encoding='utf-8') as f:
                    json_content = f.read()
                    print(json_content, file=sys.stderr)
                print("-" * 40, file=sys.stderr)

                # 显示factor_names.txt内容
                print(f"\n📋 factor_names.txt content:", file=sys.stderr)
                print("-" * 40, file=sys.stderr)
                with open(output_names_path, 'r', encoding='utf-8') as f:
                    txt_content = f.read()
                    print(txt_content, file=sys.stderr)
                print("-" * 40, file=sys.stderr)

                # 显示lorentz_config.cfg内容（模拟生成）
                print(f"\n📋 lorentz_config.cfg content (example for first date):", file=sys.stderr)
                print("-" * 40, file=sys.stderr)
                config = LorentzConfig()
                cfg_content = f"""DATE=20240101
INTERVAL_JSON={config.interval_json}

[BasicFields]
DATA_ROOT_DIR={config.data_root_dir}
LOAD_PREV_DAYS=1
THREAD_NUM={config.thread_num}
AUTO_PROD_CO_DEPENDENCY=TRUE
DAILY_DATA_DIR={config.data_root_dir}

[ComputeGraph]
THREAD_NUM={config.thread_num}
FACTOR_JSON={factor_json_path}
OUTPUT_MODULE_NAME={output_module_name}
OUTPUTS_CONFIG_FILES={output_names_path}
EMABLE_OUTPUT_CSV=TRUE
CSV_FLOAT_PRECISION=6
OUTPUT_FACTOR_ROOT_DIR={config.output_factor_root_dir}
OUTPUT_ABNORMAL_ROOT_DIR={config.output_abnormal_root_dir}"""
                print(cfg_content, file=sys.stderr)
                print("-" * 40, file=sys.stderr)

        except Exception as config_error:
            print(f"❌ Failed to generate/show config files: {config_error}", file=sys.stderr)

        # 显示原始错误
        print(f"\n💥 Original error: {str(e)}", file=sys.stderr)
        print("\n" + "="*80, file=sys.stderr)

        # 终止程序
        sys.exit(1)


def compute_batch_factor_values_with_lorentz(parsed_exprs: List[Dict]) -> Dict[str, Tuple[np.ndarray, pd.DatetimeIndex, pd.Index]]:
    """
    使用Lorentz批量计算多个因子值
    """
    config = LorentzConfig()
    executor = LorentzExecutor(config)
    parser = LorentzResultParser(config)

    # 创建debug目录
    debug_dir = os.path.join(os.getcwd(), 'lorentz_debug')
    os.makedirs(debug_dir, exist_ok=True)
    print(f"DEBUG: Created lorentz_debug directory at: {debug_dir}", file=sys.stderr)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 为批量表达式生成配置文件
        factor_json_path, output_names_path, output_module_name = generate_batch_lorentz_config_files(
            parsed_exprs, temp_dir
        )

        # 复制配置文件到debug目录
        import shutil
        debug_factor_json = os.path.join(debug_dir, 'batch_factor_config.json')
        debug_output_names = os.path.join(debug_dir, 'batch_factor_names.txt')
        shutil.copy2(factor_json_path, debug_factor_json)
        shutil.copy2(output_names_path, debug_output_names)
        print(f"DEBUG: Saved batch config files to debug directory", file=sys.stderr)

        # 解析日期范围
        start_date = datetime.strptime(config.start_date, '%Y%m%d')
        end_date = datetime.strptime(config.end_date, '%Y%m%d')

        all_results = {}

        # 计算LOAD_PREV_DAYS（与generate_batch_lorentz_config_files中的逻辑一致）
        max_prev_days = 1  # 最小值
        for parsed_expr in parsed_exprs:
            expr_str = parsed_expr["expression"]
            parsed_result = parse_expression_with_intermediates(convert_field_references(expr_str))
            all_subexpressions = []
            for intermediate in parsed_result['slice_intermediates']:
                all_subexpressions.append(intermediate['expression'])
            for intermediate in parsed_result['cross_section_intermediates']:
                all_subexpressions.append(intermediate['expression'])
            all_subexpressions.append(parsed_result['final_expression'])

            for sub_expr in all_subexpressions:
                lookback_config = analyze_lookback_requirements(sub_expr)
                if 'rolling_prev_days' in lookback_config:
                    max_prev_days = max(max_prev_days, lookback_config['rolling_prev_days'])

        load_prev_days = max_prev_days

        # 为每个日期执行批量计算
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')

            try:
                # 执行批量Lorentz计算
                success, error_msg = executor.execute_for_date(
                    date_str, factor_json_path, output_names_path, output_module_name, load_prev_days
                )

                if success:
                    # 解析批量结果
                    batch_results = parser.parse_batch_factor_output(date_str, [expr["factor_name"] for expr in parsed_exprs])
                    if batch_results:
                        for factor_name, result_df in batch_results.items():
                            if factor_name not in all_results:
                                all_results[factor_name] = []
                            all_results[factor_name].append(result_df)
                        logger.info(f"Successfully computed batch factors for {date_str}")
                    else:
                        logger.warning(f"Failed to parse batch results for {date_str}")
                else:
                    # Lorentz 执行失败，打印详细诊断信息并终止程序
                    print(f"\n" + "="*100, file=sys.stderr)
                    print(f"🚨 LORENTZ EXECUTION FAILED FOR {date_str} - TERMINATING PROGRAM", file=sys.stderr)
                    print("="*100, file=sys.stderr)

                    # 打印失败的基本信息
                    print(f"\n❌ Lorentz execution failed with error: {error_msg}", file=sys.stderr)
                    print(f"📅 Date: {date_str}", file=sys.stderr)
                    print(f"🔢 Load prev days: {load_prev_days}", file=sys.stderr)

                    # 显示配置文件内容
                    print(f"\n📋 Configuration files content:", file=sys.stderr)

                    # 显示factor_config.json
                    print(f"\n🔧 factor_config.json:", file=sys.stderr)
                    print("-" * 60, file=sys.stderr)
                    try:
                        with open(factor_json_path, 'r', encoding='utf-8') as f:
                            json_content = f.read()
                            print(json_content, file=sys.stderr)
                    except Exception as e:
                        print(f"❌ Failed to read factor_config.json: {e}", file=sys.stderr)
                    print("-" * 60, file=sys.stderr)

                    # 显示factor_names.txt
                    print(f"\n📝 factor_names.txt:", file=sys.stderr)
                    print("-" * 60, file=sys.stderr)
                    try:
                        with open(output_names_path, 'r', encoding='utf-8') as f:
                            txt_content = f.read()
                            print(txt_content, file=sys.stderr)
                    except Exception as e:
                        print(f"❌ Failed to read factor_names.txt: {e}", file=sys.stderr)
                    print("-" * 60, file=sys.stderr)

                    # 显示lorentz_config.cfg（从debug目录读取）
                    debug_dir = os.path.join(os.getcwd(), 'lorentz_debug')
                    cfg_file_path = os.path.join(debug_dir, f'lorentz_config_{date_str}.cfg')
                    print(f"\n⚙️ lorentz_config.cfg ({date_str}):", file=sys.stderr)
                    print("-" * 60, file=sys.stderr)
                    try:
                        with open(cfg_file_path, 'r', encoding='utf-8') as f:
                            cfg_content = f.read()
                            print(cfg_content, file=sys.stderr)
                    except Exception as e:
                        print(f"❌ Failed to read lorentz_config.cfg: {e}", file=sys.stderr)
                    print("-" * 60, file=sys.stderr)

                    # 显示Lorentz程序信息
                    config = LorentzConfig()
                    print(f"\n🏭 Lorentz Program Information:", file=sys.stderr)
                    print(f"   Executable: {config.lorentz_executable}", file=sys.stderr)
                    print(f"   Thread num: {config.thread_num}", file=sys.stderr)
                    print(f"   Data root: {config.data_root_dir}", file=sys.stderr)
                    print(f"   Output root: {config.output_factor_root_dir}", file=sys.stderr)

                    # 检查输入文件是否存在
                    print(f"\n📁 Input Files Check:", file=sys.stderr)
                    files_to_check = [
                        ('Interval JSON', config.interval_json),
                        ('Data Root', config.data_root_dir),
                        ('Factor JSON', factor_json_path),
                        ('Output Names', output_names_path),
                    ]

                    for name, path in files_to_check:
                        exists = os.path.exists(path)
                        status = "✅ EXISTS" if exists else "❌ MISSING"
                        print(f"   {name}: {path} - {status}", file=sys.stderr)

                        if not exists and name in ['Interval JSON', 'Data Root']:
                            print(f"      ⚠️  This is a critical file for Lorentz execution!", file=sys.stderr)

                    # 显示预期的输出目录
                    expected_output_dir = os.path.join(
                        config.output_factor_root_dir,
                        "AutoML",
                        date_str[:4],  # 年份
                        date_str      # 完整日期
                    )
                    print(f"\n📤 Expected Output Directory: {expected_output_dir}", file=sys.stderr)
                    if os.path.exists(expected_output_dir):
                        print(f"   Status: ✅ EXISTS", file=sys.stderr)
                        # 列出目录内容
                        try:
                            contents = os.listdir(expected_output_dir)
                            csv_files = [f for f in contents if f.endswith('.csv')]
                            print(f"   CSV files found: {len(csv_files)}", file=sys.stderr)
                            if csv_files:
                                print(f"   Sample files: {csv_files[:3]}", file=sys.stderr)
                        except Exception as e:
                            print(f"   Error listing directory: {e}", file=sys.stderr)
                    else:
                        print(f"   Status: ❌ DOES NOT EXIST", file=sys.stderr)

                    print(f"\n💥 TERMINATING PROGRAM DUE TO LORENTZ FAILURE", file=sys.stderr)
                    print("="*100, file=sys.stderr)

                    # 终止程序
                    sys.exit(1)

            except Exception as e:
                # 非Lorentz执行异常，打印并继续（或终止，根据严重程度）
                logger.error(f"Exception during batch Lorentz computation for {date_str}: {e}")
                import traceback
                traceback.print_exc()

                # 对于严重异常，也终止程序
                print(f"\n💥 CRITICAL EXCEPTION DURING FACTOR COMPUTATION - TERMINATING", file=sys.stderr)
                sys.exit(1)

            current_date += timedelta(days=1)

        if not all_results:
            raise ValueError("No factor values were successfully computed in batch mode")

        # 合并结果并返回
        final_results = {}
        for factor_name, result_dfs in all_results.items():
            if result_dfs:
                combined_df = pd.concat(result_dfs, ignore_index=True)
                # 转换为numpy数组格式
                pivot_df = combined_df.pivot(index='date', columns='symbol', values=factor_name)
                pivot_df = pivot_df.fillna(0.0)
                values = pivot_df.values
                dates = pivot_df.index
                symbols = pivot_df.columns
                final_results[factor_name] = (values, dates, symbols)

        return final_results


def generate_batch_lorentz_config_files(parsed_exprs: List[Dict], temp_dir: str) -> Tuple[str, str, str]:
    """
    为批量表达式生成Lorentz配置文件
    """
    # 计算LOAD_PREV_DAYS
    max_prev_days = 1  # 最小值
    has_prev_days = False

    for parsed_expr in parsed_exprs:
        expr_str = parsed_expr["expression"]

        # 解析表达式，提取所有子表达式
        parsed_result = parse_expression_with_intermediates(expr_str)
        all_subexpressions = []

        # 收集所有子表达式
        for intermediate in parsed_result['slice_intermediates']:
            all_subexpressions.append(intermediate['expression'])
        for intermediate in parsed_result['cross_section_intermediates']:
            all_subexpressions.append(intermediate['expression'])
        all_subexpressions.append(parsed_result['final_expression'])

        # 检查每个子表达式的rolling_prev_days
        for sub_expr in all_subexpressions:
            lookback_config = analyze_lookback_requirements(sub_expr)
            if 'rolling_prev_days' in lookback_config:
                max_prev_days = max(max_prev_days, lookback_config['rolling_prev_days'])
                has_prev_days = True

    # 如果没有rolling_prev_days，使用默认值1
    load_prev_days = max_prev_days if has_prev_days else 1

    # 构建批量配置
    default_config = {
        "slice": {
            "trigger": "slice",
            "output": True
        }
    }

    slice_configs = []

    # 为每个表达式创建配置，所有output都设为true
    for parsed_expr in parsed_exprs:
        factor_name = parsed_expr["factor_name"]
        expression = parsed_expr["expression"]

        # 转换字段引用
        converted_expression = convert_field_references(expression)

        # 解析表达式
        parsed_result = parse_expression_with_intermediates(converted_expression)

        # 为slice中间变量创建配置
        for intermediate in parsed_result['slice_intermediates']:
            slice_configs.append(intermediate)

        # 为最终表达式创建配置
        final_config = {
            "name": factor_name,
            "expression": parsed_result['final_expression'],
            "trigger": "slice",  # 批量模式都放在slice
            "output": True  # 所有表达式output都为true
        }

        # 添加lookback配置
        final_lookback = analyze_lookback_requirements(parsed_result['final_expression'])
        if final_lookback:
            final_config.update(final_lookback)

        slice_configs.append(final_config)

    # 构建最终JSON
    factor_json = {
        "default": default_config,
        "slice": slice_configs
    }

    factor_json_path = os.path.join(temp_dir, "batch_factor_config.json")
    with open(factor_json_path, 'w', encoding='utf-8') as f:
        json.dump(factor_json, f, indent=2, ensure_ascii=False)

    # 生成输出因子名称文件（所有因子）
    output_names = [expr["factor_name"] for expr in parsed_exprs]
    output_names_path = os.path.join(temp_dir, "batch_factor_names.txt")
    with open(output_names_path, 'w', encoding='utf-8') as f:
        for name in output_names:
            f.write(f"{name}\n")

    # 输出模块名称
    output_module_name = f"batch_set_{hash(str(output_names)) % 10000}"

    return factor_json_path, output_names_path, output_module_name


def compute_factor_values(parsed_expr: dict, data_source=None) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.Index]:
    """
    计算单个因子值的主函数（保持向后兼容）
    """
    return compute_batch_factor_values([parsed_expr], data_source)[parsed_expr["factor_name"]]


def main():
    """
    主函数：解析命令行参数并计算因子值
    """
    if len(sys.argv) != 2:
        print("Usage: python external_compute.py <alpha_expression>", file=sys.stderr)
        sys.exit(1)

    expr_str = sys.argv[1]

    try:
        # 解析表达式
        parsed_expr = parse_alpha_expression(expr_str)
        print(f"Parsed expression: {expr_str}", file=sys.stderr)

        # 计算因子值
        values, dates, symbols = compute_factor_values(parsed_expr)

        # 输出CSV格式结果
        print("date,symbol,value")

        # 遍历所有日期和股票
        for i, date in enumerate(dates):
            for j, symbol in enumerate(symbols):
                value = values[i, j]
                if not np.isnan(value):  # 只输出非NaN值
                    print(f"{date.strftime('%Y-%m-%d')},{symbol},{value:.6f}")

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

