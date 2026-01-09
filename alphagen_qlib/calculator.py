from typing import List, Optional, Dict
from torch import Tensor
import torch
import numpy as np
import sys
import threading
import time
import concurrent.futures
from collections import defaultdict
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import StockData


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression]):
        self.data = data
        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data))

    def _calc_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            ic = batch_pearsonr(ensemble_value, self.target_value).mean().item()
            return ic

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            rank_ic = batch_spearmanr(ensemble_value, self.target_value).mean().item()
            return rank_ic

class TestStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression]):
        self.data = data

        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data)).cpu().half()

    def _calc_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data)).cpu().half()

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            ic = batch_pearsonr(ensemble_value, self.target_value).mean().item()
            return ic

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            rank_ic = batch_spearmanr(ensemble_value, self.target_value).mean().item()
            return rank_ic


class ExternalCalculator(AlphaCalculator):
    def __init__(self, device: torch.device, external_func, batch_size=20):
        self.device = device
        self.external_func = external_func  # Function to call external engine
        self.target_value = None  # Not used in external calculation mode

        # 异步批量计算相关
        self.batch_size = batch_size
        self.pending_expressions = []  # 待计算的表达式队列
        self.computed_results = {}     # 已计算的结果缓存
        self.estimating_results = {}   # 临时估计结果
        self.lock = threading.Lock()   # 线程锁
        self.batch_thread = None       # 批量计算线程

        # 启动批量计算线程
        self._start_batch_thread()

    def _calc_alpha(self, expr: Expression) -> Tensor:
        # Get infix string or suitable format for external
        expr_str = str(expr)  # Or implement to_infix if needed
        # Call external function to get factor values
        values, dates, symbols = self.external_func(expr_str)
        # Assume values is (n_days, n_stocks)
        # Normalize as per original
        tensor = torch.tensor(values, dtype=torch.float, device=self.device)
        return normalize_by_day(tensor)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Expression) -> float:
        # 对所有合法表达式使用异步批量计算
        expr_str = str(expr)

        # 首先检查是否已有计算结果
        with self.lock:
            if expr_str in self.computed_results:
                return self.computed_results[expr_str]

        # 添加到待计算队列
        with self.lock:
            if expr_str not in [str(e) for e in self.pending_expressions]:
                self.pending_expressions.append(expr)

        # 返回估计值，等待批量计算完成
        return self._estimate_ic(expr)

    def _start_batch_thread(self):
        """启动异步批量计算线程"""
        def batch_worker():
            while True:
                try:
                    # 检查是否有足够的待计算表达式
                    with self.lock:
                        if len(self.pending_expressions) >= self.batch_size:
                            # 复制待计算表达式
                            batch_exprs = self.pending_expressions[:self.batch_size]
                            self.pending_expressions = self.pending_expressions[self.batch_size:]
                        else:
                            batch_exprs = []

                    if batch_exprs:
                        # 批量计算IC
                        print(f"🔄 Starting batch IC calculation for {len(batch_exprs)} expressions...")
                        batch_results = self._batch_compute_ic(batch_exprs)

                        # 更新结果缓存
                        with self.lock:
                            self.computed_results.update(batch_results)
                            print(f"✅ Batch IC calculation completed, {len(batch_results)} results cached")

                    # 短暂休眠避免CPU占用过高
                    time.sleep(0.1)

                except Exception as e:
                    print(f"❌ Error in batch worker: {e}", file=sys.stderr)
                    time.sleep(1.0)  # 出错时稍长休眠

        self.batch_thread = threading.Thread(target=batch_worker, daemon=True)
        self.batch_thread.start()

    def _batch_compute_ic(self, expressions: List[Expression]) -> Dict[str, float]:
        """批量计算多个表达式的IC"""
        results = {}

        try:
            # 使用线程池并行计算
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # 提交所有计算任务
                future_to_expr = {
                    executor.submit(self._compute_single_ic_blocking, expr): expr
                    for expr in expressions
                }

                # 收集结果
                for future in concurrent.futures.as_completed(future_to_expr):
                    expr = future_to_expr[future]
                    try:
                        ic_value = future.result()
                        expr_str = str(expr)
                        results[expr_str] = ic_value
                    except Exception as e:
                        print(f"❌ Failed to compute IC for {str(expr)}: {e}", file=sys.stderr)
                        results[str(expr)] = 0.0

        except Exception as e:
            print(f"❌ Error in batch IC computation: {e}", file=sys.stderr)
            # 返回所有表达式的默认值
            for expr in expressions:
                results[str(expr)] = 0.0

        return results

    def _compute_single_ic_blocking(self, expr: Expression) -> float:
        """阻塞式计算单个表达式的IC（用于批量计算）"""
        try:
            expr_str = str(expr)

            # 计算因子值
            values, dates, symbols = self.external_func(expr_str)

            # 加载target数据
            from adapters.scoring_calculator import target_manager
            target_data = target_manager.load_target()
            if target_data is None:
                return 0.0

            target_values = target_data['values']
            target_dates = target_data['dates']
            target_symbols = target_data['symbols']

            # 数据对齐并计算IC
            import pandas as pd
            import numpy as np

            factor_df = pd.DataFrame(values, index=dates, columns=symbols)
            target_df = pd.DataFrame(target_values, index=target_dates, columns=target_symbols)

            common_dates = factor_df.index.intersection(target_df.index)
            common_symbols = factor_df.columns.intersection(target_df.columns)

            if len(common_dates) == 0 or len(common_symbols) == 0:
                return 0.0

            aligned_factor = factor_df.loc[common_dates, common_symbols].values
            aligned_target = target_df.loc[common_dates, common_symbols].values

            # 计算IC
            from scipy.stats import pearsonr
            ic_value = pearsonr(aligned_factor.flatten(), aligned_target.flatten())[0]

            return 0.0 if np.isnan(ic_value) else ic_value

        except Exception as e:
            print(f"❌ Exception in blocking IC calculation: {e}", file=sys.stderr)
            return 0.0

    def _estimate_ic(self, expr: Expression) -> float:
        """基于表达式特征估算IC值（当真实IC还未计算完成时使用）"""
        expr_str = str(expr)

        # 检查是否已有估计值
        if expr_str in self.estimating_results:
            return self.estimating_results[expr_str]

        # 基于表达式复杂度估算IC
        # 简单表达式通常IC较低，复杂表达式可能有更高IC
        complexity = len(expr_str.split())  # 粗略的复杂度度量

        # 估算公式：复杂度贡献 + 随机噪声
        estimated_ic = min(complexity * 0.001, 0.05)  # 最大0.05
        estimated_ic += np.random.normal(0, 0.01)  # 添加噪声

        # 确保在合理范围内
        estimated_ic = max(min(estimated_ic, 0.1), -0.1)

        self.estimating_results[expr_str] = estimated_ic
        return estimated_ic

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        # For external calculator, pool IC calculation can be implemented if needed
        # For now, return 0.0 as placeholder
        return 0.0

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        # For external calculator, pool rank IC calculation can be implemented if needed
        # For now, return 0.0 as placeholder
        return 0.0
