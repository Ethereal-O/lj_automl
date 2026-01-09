"""
Alpha Scoring Calculator
用于计算已计算因子值之间的IC（信息系数）

此模块不负责因子计算，只从缓存中读取预计算的因子值来计算IC。
因子值由external_compute_factor预先计算并存储。
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import sys
import hashlib
import pickle
from pathlib import Path
import os
import time
from datetime import datetime

# 导入机器学习库
try:
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    from scipy.stats import spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn/lightgbm not available, LGB evaluation will be disabled", file=sys.stderr)

# 配置
FACTOR_CACHE_DIR = "factor_cache"
TARGET_CACHE_FILE = "target_values.pkl"

# 预测目标数据配置 - 修改为新的CSV格式
FACTOR_DATA_ROOT_DIR = os.getenv('FACTOR_DATA_ROOT_DIR', "/dfs/dataset/10-1732512661487/data/StockLabel_adj_lnret")
INTERVAL_CONFIG = os.getenv('INTERVAL_CONFIG', "1dper1d")

# 预测目标列名配置
TARGET_COLUMN = os.getenv('TARGET_COLUMN', 'yhat_raw_lnRet_t2ov_1d')  # 指定使用的预测目标列名

# 聚合收益率缓存目录
RETURN_CACHE_DIR = "return_cache"


class FactorCache:
    """因子值缓存管理器"""

    def __init__(self, cache_dir: str = FACTOR_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, expr: str) -> str:
        """生成表达式缓存键"""
        return hashlib.md5(expr.encode()).hexdigest()

    def save_factor(self, expr: str, values: np.ndarray, dates: pd.Index, symbols: pd.Index):
        """保存因子值到缓存"""
        cache_key = self._get_cache_key(expr)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        data = {
            'expression': expr,
            'values': values,
            'dates': dates,
            'symbols': symbols
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def load_factor(self, expr: str) -> Optional[Dict]:
        """从缓存加载因子值"""
        cache_key = self._get_cache_key(expr)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def has_factor(self, expr: str) -> bool:
        """检查因子是否已缓存"""
        return self.load_factor(expr) is not None


class TargetManager:
    """目标值管理器"""

    def __init__(self, target_file: str = TARGET_CACHE_FILE):
        self.target_file = Path(target_file)

    def save_target(self, values: np.ndarray, dates: pd.Index, symbols: pd.Index):
        """保存目标值"""
        data = {
            'values': values,
            'dates': dates,
            'symbols': symbols
        }

        with open(self.target_file, 'wb') as f:
            pickle.dump(data, f)

    def load_target(self) -> Optional[Dict]:
        """加载目标值"""
        if self.target_file.exists():
            with open(self.target_file, 'rb') as f:
                return pickle.load(f)
        return None


# 全局实例
factor_cache = FactorCache()
target_manager = TargetManager()


class ReturnCache:
    """聚合收益率缓存管理器"""

    def __init__(self, cache_dir: str = RETURN_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, interval_config: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        key_str = f"{interval_config}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def save_returns(self, interval_config: str, start_date: str, end_date: str,
                    values: np.ndarray, dates: pd.Index, symbols: pd.Index):
        """保存聚合收益率"""
        cache_key = self._get_cache_key(interval_config, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        data = {
            'interval_config': interval_config,
            'start_date': start_date,
            'end_date': end_date,
            'values': values,
            'dates': dates,
            'symbols': symbols
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def load_returns(self, interval_config: str, start_date: str, end_date: str) -> Optional[Dict]:
        """加载聚合收益率"""
        cache_key = self._get_cache_key(interval_config, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None


# 全局实例
return_cache = ReturnCache()


def parse_interval_config(interval_config: str) -> Tuple[str, str]:
    """
    解析interval配置，返回(因子计算间隔, 预测时间跨度)

    Args:
        interval_config: 如 "30per30", "5per5", "1dper1d", "10per30"等
                         a可以是数字(5,10,30)或"1d"，b决定收益率聚合

    Returns:
        (factor_interval, prediction_period) - factor_interval可以是数字字符串或"1d"
    """
    if 'per' not in interval_config:
        # 默认配置
        return "30", "30"  # 30分钟间隔，预测30分钟

    parts = interval_config.split('per')
    if len(parts) != 2:
        return "30", "30"

    factor_interval = parts[0]  # 保持为字符串，支持"1d"等格式
    prediction_period = parts[1]

    return factor_interval, prediction_period


def aggregate_minute_returns_for_interval(df: pd.DataFrame, factor_minute: int,
                                        prediction_period: str, date_str: str) -> pd.DataFrame:
    """
    从因子计算时刻开始，聚合未来指定时间段的分钟收益率
    智能处理数据不足的情况，动态调整预测周期

    Args:
        df: 包含分钟数据的DataFrame (symbol, minuteCode, label)
        factor_minute: 因子计算的分钟时刻
        prediction_period: 预测时间跨度 ("30", "5", "1d"等)
        date_str: 日期字符串

    Returns:
        聚合后的收益率DataFrame
    """
    try:
        # 解析预测时间跨度
        if prediction_period.endswith('d'):
            # 日级别预测：暂时返回空DataFrame
            # TODO: 实现跨天数据读取
            print(f"  Note: Day-level prediction requires cross-day data (not implemented yet)", file=sys.stderr)
            return pd.DataFrame()

        else:
            # 分钟级别预测
            requested_minutes = int(prediction_period)

            # 计算理论上的未来分钟
            theoretical_future_minutes = list(range(factor_minute + 1, factor_minute + 1 + requested_minutes))

            # 获取数据中实际可用的分钟
            available_minutes = sorted(df['minuteCode'].unique())
            available_minutes = [m for m in available_minutes if m > factor_minute]

            if not available_minutes:
                # 没有未来数据
                print(f"  Warning: No future data available after minute {factor_minute}", file=sys.stderr)
                return pd.DataFrame()

            # 找到重叠的分钟（实际可用且在理论范围内）
            max_available_minute = max(available_minutes)
            max_theoretical_minute = factor_minute + requested_minutes

            # 实际可用的未来分钟
            actual_future_minutes = [m for m in theoretical_future_minutes if m in available_minutes]

            if not actual_future_minutes:
                print(f"  Warning: No overlapping future minutes found for factor_minute {factor_minute}", file=sys.stderr)
                return pd.DataFrame()

            # 如果可用数据不足，使用所有可用数据
            if len(actual_future_minutes) < requested_minutes:
                shortage = requested_minutes - len(actual_future_minutes)
                print(f"  Note: Using {len(actual_future_minutes)} minutes instead of requested {requested_minutes} (shortage: {shortage})", file=sys.stderr)

            future_minutes = actual_future_minutes

        if not future_minutes:
            return pd.DataFrame()

        # 聚合未来分钟的收益率
        # 方法：连乘 (1 + r1) * (1 + r2) * ... - 1
        future_returns = []
        for _, row in df.iterrows():
            stock_code = row['symbol']
            stock_minute_returns = []

            # 收集该股票在未来时间段的所有分钟收益率
            for minute in future_minutes:
                minute_data = df[(df['symbol'] == stock_code) & (df['minuteCode'] == minute)]
                if not minute_data.empty:
                    stock_minute_returns.append(minute_data['label'].iloc[0])
                else:
                    # 如果缺少分钟数据，用0填充
                    stock_minute_returns.append(0.0)

            # 计算累积收益率
            if stock_minute_returns:
                cumulative_return = 1.0
                for ret in stock_minute_returns:
                    cumulative_return *= (1.0 + ret)
                cumulative_return -= 1.0
            else:
                cumulative_return = 0.0

            future_returns.append({
                'symbol': stock_code,
                'date': pd.to_datetime(date_str, format='%Y%m%d'),
                'factor_minute': factor_minute,
                'return': cumulative_return,
                'minutes_used': len(stock_minute_returns),  # 记录使用了多少分钟的数据
                'requested_minutes': requested_minutes if 'requested_minutes' in locals() else 0
            })

        return pd.DataFrame(future_returns)

    except Exception as e:
        print(f"Error aggregating returns for minute {factor_minute}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def load_target_from_csv(start_date: str, end_date: str) -> Optional[Dict]:
    """
    从CSV文件加载预测目标数据
    文件格式: /dfs/dataset/10-1732512661487/data/Stocklabel_adj_lnret/YYYY/YYYYMMDD/eHHMMSS.csv

    只处理实际存在数据的日期（通过检查是否有对应的CSV文件目录决定）

    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    Returns:
        包含values, dates, symbols的字典，以及valid_dates列表
    """
    try:
        # 检查缓存
        cached_data = return_cache.load_returns(INTERVAL_CONFIG, start_date, end_date)
        if cached_data:
            print(f"Loaded cached target data for {INTERVAL_CONFIG}", file=sys.stderr)
            return cached_data

        from datetime import datetime, timedelta

        # 解析interval配置
        factor_interval, prediction_period = parse_interval_config(INTERVAL_CONFIG)

        # 第一步：扫描数据目录，找到所有有效日期
        print(f"Scanning data directory for valid dates between {start_date} and {end_date}...", file=sys.stderr)

        valid_dates = []
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        current = start

        while current <= end:
            date_str = current.strftime('%Y%m%d')
            year = date_str[:4]
            date_dir = os.path.join(FACTOR_DATA_ROOT_DIR, year, date_str)

            # 检查该日期是否有数据目录
            if os.path.exists(date_dir):
                # 检查目录下是否有CSV文件
                try:
                    csv_files = [f for f in os.listdir(date_dir) if f.startswith('e') and f.endswith('.csv')]
                    if csv_files:  # 有CSV文件才算有效日期
                        valid_dates.append(date_str)
                        print(f"  Found valid date: {date_str} ({len(csv_files)} files)", file=sys.stderr)
                except Exception as e:
                    print(f"  Warning: Failed to scan {date_dir}: {e}", file=sys.stderr)

            current += timedelta(days=1)

        if not valid_dates:
            print(f"No valid dates found with CSV data between {start_date} and {end_date}", file=sys.stderr)
            return None

        print(f"Found {len(valid_dates)} valid dates: {valid_dates[:5]}{'...' if len(valid_dates) > 5 else ''}", file=sys.stderr)

        # 第二步：只处理有效日期的数据 (优化版本)
        all_target_data = []

        for date_str in valid_dates:
            year = date_str[:4]
            date_dir = os.path.join(FACTOR_DATA_ROOT_DIR, year, date_str)

            try:
                print(f"Processing {date_str}: loading CSV files...", file=sys.stderr)
                start_time = time.time()

                # 获取该日期目录下的所有CSV文件
                csv_files = [f for f in os.listdir(date_dir) if f.startswith('e') and f.endswith('.csv')]
                print(f"  Found {len(csv_files)} CSV files", file=sys.stderr)

                # 预解析所有文件的时间戳
                file_info_list = []
                for csv_file in csv_files:
                    time_str = csv_file[1:-4]  # 去掉'e'和'.csv'
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4])
                    second = int(time_str[4:])

                    # 计算从9:30开始的分钟偏移量
                    base_hour, base_minute = 9, 30
                    total_minutes = (hour - base_hour) * 60 + (minute - base_minute)
                    if total_minutes < 0:  # 处理跨天情况
                        total_minutes += 24 * 60

                    file_info_list.append({
                        'path': os.path.join(date_dir, csv_file),
                        'factor_minute': total_minutes
                    })

                # 批量读取和处理所有CSV文件
                batch_data = []
                for file_info in file_info_list:
                    try:
                        csv_path = file_info['path']
                        factor_minute = file_info['factor_minute']

                        # 使用更高效的参数读取CSV
                        df = pd.read_csv(csv_path, usecols=['skey', TARGET_COLUMN, 'isZT', 'isDT'] if 'isZT' in pd.read_csv(csv_path, nrows=0).columns else ['skey', TARGET_COLUMN])

                        # 过滤涨跌停数据（如果列存在）
                        if 'isZT' in df.columns and 'isDT' in df.columns:
                            df = df[(df['isZT'] == 0) & (df['isDT'] == 0)]

                        # 向量化操作：直接构建数据
                        date_timestamp = pd.to_datetime(date_str, format='%Y%m%d')

                        # 使用向量化方式构建数据框
                        temp_df = pd.DataFrame({
                            'symbol': df['skey'],
                            'date': date_timestamp,
                            'factor_minute': factor_minute,
                            'return': df[TARGET_COLUMN]
                        })

                        batch_data.append(temp_df)

                    except Exception as e:
                        print(f"Warning: Failed to process {csv_path}: {e}", file=sys.stderr)
                        continue

                # 批量合并该日期的所有数据
                if batch_data:
                    try:
                        date_combined = pd.concat(batch_data, ignore_index=True)
                        # 确保DataFrame有正确的列
                        required_columns = ['symbol', 'date', 'factor_minute', 'return']
                        if all(col in date_combined.columns for col in required_columns):
                            all_target_data.append(date_combined)
                        else:
                            print(f"  Warning: Missing required columns for {date_str}", file=sys.stderr)
                    except Exception as e:
                        print(f"  Warning: Failed to merge data for {date_str}: {e}", file=sys.stderr)

                    processing_time = time.time() - start_time
                    print(f"  Completed processing {date_str} in {processing_time:.2f}s, collected {len(date_combined)} records", file=sys.stderr)
                else:
                    print(f"  Warning: No data collected for {date_str}", file=sys.stderr)

            except Exception as e:
                print(f"Warning: Failed to process date {date_str}: {e}", file=sys.stderr)
                continue

        if not all_target_data:
            print("No target data collected from valid dates", file=sys.stderr)
            return None

        # 合并所有数据
        if all_target_data:
            combined_df = pd.concat(all_target_data, ignore_index=True)
            print(f"Total collected {len(combined_df)} records from {len(valid_dates)} dates", file=sys.stderr)
        else:
            print("No target data collected from valid dates", file=sys.stderr)
            return None

        # 处理数据聚合
        if factor_interval == "1d":
            # 日级别：按日期聚合，合并所有时间点的数据
            pivot_df = combined_df.pivot_table(
                index='date',
                columns='symbol',
                values='return',
                aggfunc='mean'  # 对同一天多个时间点取平均
            )
        else:
            # 分钟级别：按日期+因子分钟聚合
            pivot_df = combined_df.pivot_table(
                index=['date', 'factor_minute'],
                columns='symbol',
                values='return',
                aggfunc='first'  # 每个时间点应该只有一个值
            )

        # 填充缺失值
        pivot_df = pivot_df.fillna(0.0)

        # 转换为numpy数组
        values = pivot_df.values
        dates = pivot_df.index
        symbols = pivot_df.columns

        # 保存到缓存
        return_cache.save_returns(INTERVAL_CONFIG, start_date, end_date, values, dates, symbols)

        print(f"Loaded target data: {values.shape} using {INTERVAL_CONFIG} from column '{TARGET_COLUMN}'", file=sys.stderr)
        print(f"Valid dates processed: {len(valid_dates)} out of requested range", file=sys.stderr)

        return {
            'values': values,
            'dates': dates,
            'symbols': symbols,
            'valid_dates': valid_dates  # 返回有效日期列表
        }

    except Exception as e:
        print(f"Error loading target from CSV: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


# 为了向后兼容，重命名函数
load_target_from_parquet = load_target_from_csv





def calculate_ic_from_values(alpha_values: np.ndarray, target_values: np.ndarray) -> float:
    """
    从因子值数组计算IC

    Args:
        alpha_values: alpha因子值数组 (n_days, n_stocks)
        target_values: 目标值数组 (n_days, n_stocks)

    Returns:
        IC值 (float): 皮尔逊相关系数
    """
    try:
        # 展平数组
        alpha_flat = alpha_values.flatten()
        target_flat = target_values.flatten()

        # 移除NaN值
        mask = ~(np.isnan(alpha_flat) | np.isnan(target_flat))
        alpha_clean = alpha_flat[mask]
        target_clean = target_flat[mask]

        if len(alpha_clean) == 0:
            return 0.0

        # 计算相关系数
        correlation_matrix = np.corrcoef(alpha_clean, target_clean)
        ic = correlation_matrix[0, 1]

        # 检查是否为NaN
        return 0.0 if np.isnan(ic) else ic

    except Exception as e:
        print(f"Error calculating IC from values: {e}", file=sys.stderr)
        return 0.0


def calculate_alpha_ic(alpha_expr: str, target_expr: str = None) -> float:
    """
    计算alpha表达式与目标的IC
    从已缓存的因子数据中读取计算

    Args:
        alpha_expr: Alpha表达式字符串
        target_expr: 目标表达式字符串（可选，用于指定特定目标）

    Returns:
        IC值 (float): 信息系数
    """
    try:
        print(f"🔍 calculate_alpha_ic called for: {alpha_expr}", file=sys.stderr)

        # 直接从缓存加载alpha因子值（已由external_compute_factor预先计算并缓存）
        alpha_data = factor_cache.load_factor(alpha_expr)
        if alpha_data is None:
            print(f"❌ Alpha factor not found in cache: {alpha_expr}", file=sys.stderr)
            print("💡 Make sure external_compute_factor has been called for this expression first", file=sys.stderr)
            return 0.0

        print(f"✅ Found cached alpha data: shape {alpha_data['values'].shape}", file=sys.stderr)

        alpha_values = alpha_data['values']

        # 加载目标值
        if target_expr:
            # 使用指定的目标表达式
            target_data = factor_cache.load_factor(target_expr)
            if target_data is None:
                print(f"Target factor not found in cache: {target_expr}", file=sys.stderr)
                return 0.0
            target_values = target_data['values']
        else:
            # 使用默认目标（未来收益率）
            target_data = target_manager.load_target()
            if target_data is None:
                # 尝试从parquet文件加载
                print("Loading target from parquet files...", file=sys.stderr)
                # 从alpha数据获取日期范围
                alpha_dates = alpha_data['dates']
                if len(alpha_dates) > 0:
                    start_date = alpha_dates.min().strftime('%Y%m%d')
                    end_date = alpha_dates.max().strftime('%Y%m%d')
                    target_data = load_target_from_parquet(start_date, end_date)

                if target_data is None:
                    print("Default target values not found. Make sure target has been pre-computed or parquet files are available.", file=sys.stderr)
                    return 0.0

            target_values = target_data['values']

        # 确保维度匹配
        if alpha_values.shape != target_values.shape:
            print(f"Shape mismatch: alpha {alpha_values.shape} vs target {target_values.shape}", file=sys.stderr)
            return 0.0

        # 计算IC
        ic = calculate_ic_from_values(alpha_values, target_values)
        return ic

    except Exception as e:
        print(f"Error calculating IC for expression {alpha_expr}: {e}", file=sys.stderr)
        return 0.0


def evaluate_factor_combination_lgb(exprs: list, baseline_exprs: list = None) -> dict:
    """
    使用LightGBM评估因子组合的预测能力 (类似lgb_baseline的方式)
    返回详细的评估指标，主要关注q5表现

    Args:
        exprs: 新增的表达式列表
        baseline_exprs: 基准表达式列表（已有因子）

    Returns:
        评估指标字典，包含IC, RankIC, q5收益, q1收益等
    """
    try:
        from sklearn.model_selection import train_test_split
        import lightgbm as lgb
        from scipy.stats import spearmanr

        # 加载所有因子数据
        all_exprs = (baseline_exprs or []) + exprs
        if not all_exprs:
            return {"error": "No expressions provided"}

        factor_data_list = []
        for expr in all_exprs:
            data = factor_cache.load_factor(expr)
            if data is None:
                return {"error": f"Factor not found: {expr}"}
            factor_data_list.append(data)

        # 确保所有因子有相同的维度
        base_shape = factor_data_list[0]['values'].shape
        if not all(data['values'].shape == base_shape for data in factor_data_list):
            # 如果维度不匹配，尝试按日期对齐
            print(f"⚠️ Factor dimensions don't match, attempting to align by date...")
            try:
                aligned_data = []
                base_dates = factor_data_list[0]['dates']
                base_symbols = factor_data_list[0]['symbols']

                for data in factor_data_list:
                    # 对齐日期和股票
                    common_dates = base_dates.intersection(data['dates'])
                    common_symbols = base_symbols.intersection(data['symbols'])

                    if len(common_dates) == 0 or len(common_symbols) == 0:
                        return {"error": f"Cannot align factor data: no common dates/symbols"}

                    # 重新索引数据
                    df = pd.DataFrame(data['values'], index=data['dates'], columns=data['symbols'])
                    aligned_df = df.loc[common_dates, common_symbols]

                    aligned_data.append({
                        'values': aligned_df.values,
                        'dates': aligned_df.index,
                        'symbols': aligned_df.columns
                    })

                factor_data_list = aligned_data
                base_shape = factor_data_list[0]['values'].shape
                print(f"✅ Successfully aligned factor data to shape: {base_shape}")

            except Exception as e:
                return {"error": f"Failed to align factor dimensions: {str(e)}"}

        # 合并因子数据作为特征
        X = np.stack([data['values'] for data in factor_data_list], axis=-1)  # (n_days, n_stocks, n_factors)
        n_days, n_stocks, n_factors = X.shape

        # 加载目标值并确保维度匹配
        target_data = target_manager.load_target()
        if target_data is None:
            return {"error": "Target values not found"}

        # 检查目标数据维度是否与因子数据匹配
        target_values = target_data['values']
        target_dates = target_data['dates']
        target_symbols = target_data['symbols']

        factor_dates = factor_data_list[0]['dates']
        factor_symbols = factor_data_list[0]['symbols']

        # 如果维度不匹配，尝试对齐
        if target_values.shape != (n_days, n_stocks):
            print(f"⚠️ Target shape {target_values.shape} doesn't match factors {(n_days, n_stocks)}, attempting alignment...")

            # 创建目标数据DataFrame
            target_df = pd.DataFrame(target_values, index=target_dates, columns=target_symbols)

            # 对齐到因子数据的日期和股票
            common_dates = factor_dates.intersection(target_dates)
            common_symbols = factor_symbols.intersection(target_symbols)

            if len(common_dates) == 0 or len(common_symbols) == 0:
                return {"error": "Cannot align target data with factor data: no common dates/symbols"}

            aligned_target_df = target_df.loc[common_dates, common_symbols]
            y = aligned_target_df.values

            # 同时调整因子数据的维度
            factor_df = pd.DataFrame(X[:, :, 0], index=factor_dates, columns=factor_symbols)
            aligned_factor_df = factor_df.loc[common_dates, common_symbols]

            n_days, n_stocks = aligned_factor_df.shape
            X = np.stack([aligned_factor_df.values] + [data['values'] for data in factor_data_list[1:]], axis=-1)

            print(f"✅ Successfully aligned data to shape: factors {(n_days, n_stocks, n_factors)}, target {(n_days, n_stocks)}")
        else:
            y = target_values  # (n_days, n_stocks)

        # 展平数据用于LightGBM
        X_flat = X.reshape(n_days * n_stocks, n_factors)
        y_flat = y.reshape(n_days * n_stocks)

        # 移除NaN值
        valid_mask = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y_flat))
        X_flat = X_flat[valid_mask]
        y_flat = y_flat[valid_mask]

        if len(X_flat) < 1000:  # 样本太少
            return {"error": "Insufficient data for evaluation"}

        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y_flat, test_size=0.2, random_state=42
        )

        # 训练LightGBM模型
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': min(31, max(10, n_factors * 2)),
            'learning_rate': 0.05,
            'feature_fraction': min(0.9, max(0.5, n_factors / 20.0)),
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4
        }

        model = lgb.train(params, train_data, num_boost_round=100)

        # 预测
        y_pred = model.predict(X_test)

        # 计算基础指标
        ic = calculate_ic_from_values(y_pred.reshape(-1, 1), y_test.reshape(-1, 1))
        rank_ic = spearmanr(y_pred, y_test)[0] if len(y_pred) > 10 else 0.0

        # 计算q5-q1收益差异 (核心指标)
        n_samples = len(y_pred)
        if n_samples >= 20:  # 至少需要20个样本
            # 按预测值排序
            sorted_indices = np.argsort(y_pred)

            # q5: 最好的20%
            q5_indices = sorted_indices[int(0.8 * n_samples):]
            q5_return = np.mean(y_test[q5_indices])

            # q1: 最差的20%
            q1_indices = sorted_indices[:int(0.2 * n_samples)]
            q1_return = np.mean(y_test[q1_indices])

            # q5-q1差异 (bps)
            q5_q1_diff = (q5_return - q1_return) * 10000  # 转换为bps

            # 分位数收益
            q5_return_bps = q5_return * 10000
            q1_return_bps = q1_return * 10000

        else:
            q5_return_bps = q1_return_bps = q5_q1_diff = 0.0

        # 计算因子重要性 (如果有新增因子)
        feature_importance = {}
        if exprs and baseline_exprs:
            # 计算新增因子的边际贡献
            try:
                importance = model.feature_importance(importance_type='gain')
                baseline_count = len(baseline_exprs)

                # 新增因子的平均重要性
                if len(exprs) > 0:
                    new_factor_importance = np.mean(importance[baseline_count:])
                    baseline_importance = np.mean(importance[:baseline_count]) if baseline_count > 0 else 0
                    importance_ratio = new_factor_importance / max(baseline_importance, 1e-6)
                    feature_importance = {
                        "new_factor_avg_importance": new_factor_importance,
                        "baseline_avg_importance": baseline_importance,
                        "importance_ratio": importance_ratio
                    }
            except:
                pass

        return {
            "ic": ic if not np.isnan(ic) else 0.0,
            "rank_ic": rank_ic if not np.isnan(rank_ic) else 0.0,
            "q5_return_bps": q5_return_bps,
            "q1_return_bps": q1_return_bps,
            "q5_q1_diff_bps": q5_q1_diff,  # 核心指标
            "n_factors": n_factors,
            "n_samples": len(X_flat),
            "feature_importance": feature_importance
        }

    except Exception as e:
        return {"error": f"Error in LGB evaluation: {str(e)}"}


def calculate_factor_reward(metrics: dict, prev_metrics: dict = None) -> float:
    """
    根据评估指标计算RL奖励

    Args:
        metrics: 当前因子组合的评估指标
        prev_metrics: 之前的评估指标（用于计算增量）

    Returns:
        奖励值
    """
    if "error" in metrics:
        return -1.0  # 错误情况给予惩罚

    # 主要关注q5收益表现
    q5_reward = metrics.get("q5_return_bps", 0.0) * 0.01  # q5收益权重

    # IC的贡献 (辅助指标)
    ic_reward = metrics.get("ic", 0.0) * 0.5

    # q5-q1差异的贡献 (选股能力)
    q5_q1_reward = max(0, metrics.get("q5_q1_diff_bps", 0.0)) * 0.005

    # 因子重要性奖励 (新增因子的贡献)
    importance_reward = 0.0
    if "feature_importance" in metrics:
        importance_ratio = metrics["feature_importance"].get("importance_ratio", 1.0)
        if importance_ratio > 1.2:  # 新因子明显更有贡献
            importance_reward = (importance_ratio - 1.0) * 0.1

    # 基础奖励
    base_reward = q5_reward + ic_reward + q5_q1_reward + importance_reward

    # 如果有历史数据，计算增量奖励
    if prev_metrics and "error" not in prev_metrics:
        prev_q5 = prev_metrics.get("q5_return_bps", 0.0)
        current_q5 = metrics.get("q5_return_bps", 0.0)
        q5_increment = current_q5 - prev_q5

        # 增量奖励 (更大的权重)
        increment_reward = q5_increment * 0.02

        return base_reward + increment_reward

    return base_reward


def calculate_pool_ic_lgb(exprs: list, baseline_exprs: list = None) -> float:
    """
    兼容性函数：返回奖励值用于AlphaPool
    """
    metrics = evaluate_factor_combination_lgb(exprs, baseline_exprs)
    return calculate_factor_reward(metrics)


def calculate_pool_ic(exprs: list, weights: list) -> float:
    """
    计算因子池的组合IC (传统方法)

    Args:
        exprs: 表达式列表
        weights: 权重列表

    Returns:
        组合IC值
    """
    try:
        if not exprs or not weights:
            return 0.0

        # 加载所有因子值
        factor_values = []
        for expr in exprs:
            data = factor_cache.load_factor(expr)
            if data is None:
                print(f"Factor not found: {expr}", file=sys.stderr)
                return 0.0
            factor_values.append(data['values'])

        # 计算加权组合
        combined_values = np.zeros_like(factor_values[0])
        for values, weight in zip(factor_values, weights):
            combined_values += values * weight

        # 加载目标值
        target_data = target_manager.load_target()
        if target_data is None:
            print("Target values not found", file=sys.stderr)
            return 0.0

        target_values = target_data['values']

        # 计算组合IC
        ic = calculate_ic_from_values(combined_values, target_values)
        return ic

    except Exception as e:
        print(f"Error calculating pool IC: {e}", file=sys.stderr)
        return 0.0


def main():
    """
    主函数：从命令行参数接收表达式并计算IC
    用法：
    python scoring_calculator.py <alpha_expression> [target_expression]
    python scoring_calculator.py --pool <expr1> <weight1> <expr2> <weight2> ...
    """
    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  Single IC: python scoring_calculator.py <alpha_expression> [target_expression]", file=sys.stderr)
        print("  Pool IC: python scoring_calculator.py --pool <expr1> <weight1> <expr2> <weight2> ...", file=sys.stderr)
        sys.exit(1)

    if sys.argv[1] == '--pool':
        # 计算pool IC
        if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
            print("Pool mode requires pairs of expression and weight", file=sys.stderr)
            sys.exit(1)

        exprs = []
        weights = []
        for i in range(2, len(sys.argv), 2):
            exprs.append(sys.argv[i])
            weights.append(float(sys.argv[i + 1]))

        ic = calculate_pool_ic(exprs, weights)
    else:
        # 计算单个IC
        alpha_expr = sys.argv[1]
        target_expr = sys.argv[2] if len(sys.argv) > 2 else None
        ic = calculate_alpha_ic(alpha_expr, target_expr)

    print(f"{ic:.6f}")


if __name__ == "__main__":
    main()
