from itertools import count
import math
from typing import List, Optional, Tuple, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import Tensor
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import masked_mean_std
from alphagen_qlib.stock_data import StockData

# 导入因子缓存
from adapters.scoring_calculator import factor_cache


class AlphaPoolBase(metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        device: torch.device = torch.device('cpu')
    ):
        self.capacity = capacity
        self.calculator = calculator
        self.device = device

    @abstractmethod
    def to_dict(self) -> dict: ...

    @abstractmethod
    def try_new_expr(self, expr: Expression) -> float: ...

    @abstractmethod
    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]: ...


class AlphaPool(AlphaPoolBase):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device('cpu'),
        enable_culling: bool = False,  # Whether to enable pool culling
        culling_method: str = 'ic_drop',  # 'ic_drop', 'weight', or 'combined'
        baseline_expressions: Optional[List[str]] = None,  # 基准因子表达式列表
        use_lgb_evaluation: bool = False,  # 是否使用LightGBM评估组合效果
        reeval_cycle: int = 1000,  # 重新评估周期（每多少个因子）
        reeval_q5_threshold: float = 0.5  # q5提升阈值（bps）
    ):
        super().__init__(capacity, calculator, device)

        self.size: int = 0
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self.mutual_ics: np.ndarray = np.identity(capacity + 1)
        self.weights: np.ndarray = np.zeros(capacity + 1)
        self.best_ic_ret: float = -1.

        self.ic_lower_bound = ic_lower_bound or -1.
        self.l1_alpha = l1_alpha
        self.enable_culling = enable_culling
        self.culling_method = culling_method
        self.baseline_expressions = baseline_expressions or []
        self.use_lgb_evaluation = use_lgb_evaluation

        # 新增：重新评估相关参数
        self.reeval_cycle = reeval_cycle
        self.reeval_q5_threshold = reeval_q5_threshold

        self.eval_cnt = 0
        self._prev_metrics = None  # 存储上一次的评估指标

        # 四池管理系统
        self.premium_factors: List[Tuple[str, float]] = []  # 高贵因子：(表达式, IC)
        self.lgb_factors: List[Tuple[str, float]] = []      # LGB因子：(表达式, 组合贡献)
        self.staged_factors: List[Tuple[str, float]] = []   # 暂存因子：(表达式, 单因子IC)
        self.discarded_factors: List[Tuple[str, float]] = [] # 丢弃因子：(表达式, 单因子IC)

        # 池子容量
        self.premium_pool_capacity = capacity // 2  # 高贵因子池容量
        self.lgb_pool_capacity = capacity          # LGB因子池容量
        self.staged_pool_capacity = capacity       # 暂存因子池容量

        # 高贵因子管理
        self.pending_premium_factors: List[Tuple[str, float]] = []  # 新增高贵因子，暂时不参与LGB
        self.active_premium_factors: List[Tuple[str, float]] = []   # 参与LGB计算的高贵因子
        self.staged_cleanup_count = 0  # 暂存池清理次数计数器
        self.premium_graduation_threshold = 5  # 每5次暂存池清理，更新一次高贵因子参与状态

        # LGB评估相关
        self.current_lgb_baseline = None  # 当前LGB baseline metrics
        self.staged_evaluation_multiplier = 2.0  # 暂存因子成功加入的奖励乘数
        self._pending_staged_rewards = {}  # 暂存因子的基础奖励记录
        self._staged_episode_info = {}  # 暂存因子的episode信息，用于延迟奖励
        self._resolved_rewards = {}  # 已解决的延迟奖励

        # 评估计数器
        self.last_reeval_cnt = 0
        self.last_premium_update = 0  # 高贵因子更新计数器
        self.premium_update_cycle = 1000  # 高贵因子更新周期

    @property
    def state(self) -> dict:
        return {
            "exprs": list(self.exprs[:self.size]),
            "ics_ret": list(self.single_ics[:self.size]),
            "weights": list(self.weights[:self.size]),
            "best_ic_ret": self.best_ic_ret
        }

    def to_dict(self) -> dict:
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights[:self.size])
        }

    def try_new_expr(self, expr: Expression) -> float:
        """尝试添加新表达式，返回奖励值"""
        expr_str = str(expr)
        self.eval_cnt += 1

        # ===== 检查是否处于预热阶段 =====
        # 如果有agent引用且memory未满，直接返回0奖励，不进行IC计算
        if hasattr(self.calculator, '_agent_ref') and self.calculator._agent_ref():
            agent = self.calculator._agent_ref()
            # 检查多种memory表示方式
            memory_size = 0
            if hasattr(agent, 'memory'):
                if hasattr(agent.memory, 'size'):
                    memory_size = agent.memory.size()
                elif hasattr(agent.memory, '__len__'):
                    memory_size = len(agent.memory)
                elif hasattr(agent.memory, '_buffer'):
                    memory_size = len(agent.memory._buffer) if hasattr(agent.memory, '_buffer') else 0

            if memory_size < 10000:
                # 预热阶段：memory未满，不计算真实IC，直接返回0
                return 0.0

        # ===== 第一步：异步计算单因子IC =====
        try:
            single_ic = self.calculator.calc_single_IC_ret(expr)
            ic_threshold = max(self.ic_lower_bound, 0.01)  # 最低IC阈值
            passes_single_test = not np.isnan(single_ic) and abs(single_ic) >= ic_threshold
        except Exception as e:
            print(f"Error calculating single IC for {expr_str}: {e}")
            single_ic = 0.0
            passes_single_test = False

        # ===== 第二步：检查是否已经在LGB池中（奖励膨胀）=====
        is_in_lgb_pool = any(expr == expr_str for expr, _ in self.lgb_factors)

        # ===== 第三步：根据IC分流到不同池子 =====
        if passes_single_test:
            # 🎯 高贵因子：单因子IC足够高
            self.pending_premium_factors.append((expr_str, single_ic))
            reward = self._calculate_ic_reward(single_ic)  # 单因子IC奖励
            # 如果已经在LGB池中，奖励膨胀
            if is_in_lgb_pool:
                reward *= self.staged_evaluation_multiplier

        else:
            # 📦 暂存因子：等待组合评估，奖励延迟确定
            self.staged_factors.append((expr_str, single_ic))

            # 暂存因子episode结束时不给予奖励（延迟确定）
            reward = 0.0  # 不用于网络更新

            # 记录暂存因子的episode信息，用于后续奖励确定
            episode_id = f"episode_{self.eval_cnt}"
            self._staged_episode_info[episode_id] = {
                'expr_str': expr_str,
                'single_ic': single_ic,
                'base_reward': self._calculate_ic_reward(single_ic),
                'status': 'pending'  # pending, discarded, promoted
            }

        # ===== 第三步：处理高贵因子加入 =====
        self._process_pending_premium_factors()

        # ===== 第四步：检查暂存池是否需要评估 =====
        available_slots = self.staged_pool_capacity - len(self.staged_factors)
        if available_slots <= 0:
            # 暂存池满了，开始评估
            self._evaluate_staged_factors()

        # ===== 第五步：定期更新高贵因子参与状态 =====
        if self.eval_cnt - self.last_premium_update >= self.premium_update_cycle:
            self._update_premium_participation()

        return reward

    def _calculate_ic_reward(self, ic: float) -> float:
        """计算基于单因子IC的奖励（非线性放大高IC）"""
        return max(ic ** 2 * 2.0, 0.0)  # IC的平方作为奖励，更偏好高IC

    def _calculate_staged_reward(self, base_ic: float) -> float:
        """计算暂存因子成功加入的奖励（带乘数）"""
        return self._calculate_ic_reward(base_ic) * self.staged_evaluation_multiplier

    def _find_worst_factor_idx(self) -> int:
        """找到最差因子的索引"""
        if self.culling_method == 'weight':
            return np.argmin(np.abs(self.weights[:self.size]))
        elif self.culling_method == 'ic':
            return np.argmin(np.abs(self.single_ics[:self.size]))
        else:  # 默认使用权重
            return np.argmin(np.abs(self.weights[:self.size]))

    def _process_pending_premium_factors(self):
        """处理等待加入的高贵因子"""
        if not self.pending_premium_factors:
            return

        # 直接加入高贵因子池（所有通过筛选的都保留）
        for expr_str, ic in self.pending_premium_factors:
            self.premium_factors.append((expr_str, ic))


        self.pending_premium_factors.clear()

    def _evaluate_staged_factors(self):
        """评估暂存因子池，使用LGB评估组合效果"""
        from adapters.scoring_calculator import evaluate_factor_combination_lgb

        if not self.staged_factors:
            return

        # 获取当前参与LGB计算的因子：参与计算的高贵因子 + LGB因子
        active_premium_exprs = [expr for expr, _ in self.active_premium_factors]  # 当前参与LGB的高贵因子
        lgb_exprs = [expr for expr, _ in self.lgb_factors]

        # 构建当前baseline：参与计算的高贵因子 + LGB因子
        baseline_exprs = active_premium_exprs + lgb_exprs

        # 确保所有baseline因子的因子值都已计算
        for expr_str in baseline_exprs:
            if not factor_cache.has_factor(expr_str):
                continue

        # 评估当前baseline的性能
        baseline_metrics = evaluate_factor_combination_lgb(baseline_exprs, [])
        if "error" in baseline_metrics:
            return

        baseline_q5 = baseline_metrics.get("q5_return_bps", 0.0)

        # 逐个评估暂存因子
        promoted_factors = []
        discarded_factors = []

        for expr_str, single_ic in self.staged_factors:
            # 确保暂存因子的因子值已计算
            if not factor_cache.has_factor(expr_str):
                discarded_factors.append((expr_str, single_ic))
                continue

            # 评估加入这个暂存因子后的组合性能
            test_exprs = baseline_exprs + [expr_str]
            test_metrics = evaluate_factor_combination_lgb(test_exprs, [])

            if "error" in test_metrics:
                discarded_factors.append((expr_str, single_ic))
                continue

            combined_q5 = test_metrics.get("q5_return_bps", 0.0)
            q5_improvement = combined_q5 - baseline_q5

            # 如果q5提升超过阈值，加入LGB因子池
            if q5_improvement >= self.reeval_q5_threshold:
                promoted_factors.append((expr_str, q5_improvement))
            else:
                discarded_factors.append((expr_str, single_ic))

        # 处理提升的因子 - 奖励确定为高奖励
        for expr_str, improvement in promoted_factors:
            self.lgb_factors.append((expr_str, improvement))

            # 确定对应episode的奖励：比高贵因子更好的奖励
            for episode_id, episode_info in self._staged_episode_info.items():
                if episode_info['expr_str'] == expr_str and episode_info['status'] == 'pending':
                    # 成功加入LGB池：获得高额奖励
                    promoted_reward = self._calculate_staged_reward(episode_info['single_ic'])
                    episode_info['final_reward'] = promoted_reward
                    episode_info['status'] = 'promoted'
                    self._resolved_rewards[episode_id] = promoted_reward
                    print(f"🏆 Episode {episode_id} reward resolved: {promoted_reward:.4f} (promoted to LGB)")

            # LGB池容量管理
            if len(self.lgb_factors) > self.lgb_pool_capacity:
                self.lgb_factors.sort(key=lambda x: x[1])  # 按贡献排序
                removed = self.lgb_factors.pop(0)
                print(f"🗑️ Removed weakest LGB factor: {removed[0]}")

        # 处理丢弃的因子 - 奖励确定为普通IC奖励
        for expr_str, single_ic in discarded_factors:
            # 确定对应episode的奖励：和高贵因子一样的奖励
            for episode_id, episode_info in self._staged_episode_info.items():
                if episode_info['expr_str'] == expr_str and episode_info['status'] == 'pending':
                    # 被丢弃：获得普通IC奖励
                    discarded_reward = self._calculate_ic_reward(single_ic)
                    episode_info['final_reward'] = discarded_reward
                    episode_info['status'] = 'discarded'
                    self._resolved_rewards[episode_id] = discarded_reward
                    print(f"🗑️ Episode {episode_id} reward resolved: {discarded_reward:.4f} (discarded)")

        # 记录丢弃的因子到丢弃池
        self.discarded_factors.extend(discarded_factors)

        # 暂存池清理计数器+1
        self.staged_cleanup_count += 1

        # 清空暂存池
        self.staged_factors.clear()

        # 检查是否需要更新高贵因子参与状态
        if self.staged_cleanup_count % self.premium_graduation_threshold == 0:
            self._update_premium_participation()

    def _update_premium_participation(self):
        """定期更新高贵因子参与LGB计算的状态"""
        print(f"🔄 Updating premium factor participation status...")

        # 所有高贵因子都开始参与LGB计算（"毕业"）
        self.active_premium_factors = self.premium_factors.copy()

        print(f"✅ All {len(self.active_premium_factors)} premium factors now active in LGB evaluation")

        # 重新计算LGB baseline：新加入的高贵 + 旧高贵（已参与的） + 最新LGB因子
        from adapters.scoring_calculator import evaluate_factor_combination_lgb

        baseline_exprs = [expr for expr, _ in self.active_premium_factors + self.lgb_factors]

        # 确保所有baseline因子的因子值都已计算
        for expr_str in baseline_exprs:
            if not factor_cache.has_factor(expr_str):
                print(f"⚠️  Premium factor {expr_str} not in cache, skipping...")
                continue

        # 重新评估baseline
        baseline_metrics = evaluate_factor_combination_lgb(baseline_exprs, [])
        if "error" in baseline_metrics:
            print(f"❌ Failed to update baseline: {baseline_metrics.get('error', 'Unknown error')}")
        else:
            self.current_lgb_baseline = baseline_metrics
            baseline_q5 = baseline_metrics.get("q5_return_bps", 0.0)
            print(f"📊 Updated LGB baseline q5: {baseline_q5:.2f} bps")

        self.last_premium_update = self.eval_cnt

    def _add_factor_to_pool(self, expr: Expression):
        """添加因子到池子（简化版，不计算IC）"""
        if self.size >= self.capacity:
            if self.enable_culling:
                self._pop()
            else:
                return  # 池子已满，不添加

        n = self.size
        self.exprs[n] = expr
        self.single_ics[n] = 0.0  # 暂时设为0
        for i in range(n):
            self.mutual_ics[i][n] = self.mutual_ics[n][i] = 0.0
        self.weights[n] = 1.0  # 暂时设为1
        self.size += 1

    def force_load_exprs(self, exprs: List[Expression]) -> None:
        for expr in exprs:
            ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, ic_ret, ic_mut)
            assert self.size <= self.capacity
        self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=500)

    def _optimize(self, alpha: float, lr: float, n_iter: int) -> np.ndarray:
        if math.isclose(alpha, 0.): # no L1 regularization
            return self._optimize_lstsq() # very fast

        ics_ret = torch.from_numpy(self.single_ics[:self.size]).to(self.device)
        ics_mut = torch.from_numpy(self.mutual_ics[:self.size, :self.size]).to(self.device)
        weights = torch.from_numpy(self.weights[:self.size]).to(self.device).requires_grad_()
        optim = torch.optim.Adam([weights], lr=lr)

        loss_ic_min = 1e9 + 7  # An arbitrary big value
        best_weights = weights.cpu().detach().numpy()
        iter_cnt = 0
        for it in count():
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_ic_curr = loss_ic.item()

            loss_l1 = torch.norm(weights, p=1)  # type: ignore
            loss = loss_ic + alpha * loss_l1

            optim.zero_grad()
            loss.backward()
            optim.step()

            if loss_ic_min - loss_ic_curr > 1e-6:
                iter_cnt = 0
            else:
                iter_cnt += 1

            if loss_ic_curr < loss_ic_min:
                best_weights = weights.cpu().detach().numpy()
                loss_ic_min = loss_ic_curr

            if iter_cnt >= n_iter or it >= 10000:
                break

        return best_weights

    def _optimize_lstsq(self) -> np.ndarray:
        try:
            return np.linalg.lstsq(self.mutual_ics[:self.size, :self.size],self.single_ics[:self.size])[0]
        except (np.linalg.LinAlgError, ValueError):
            return self.weights[:self.size]

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        ic = calculator.calc_pool_IC_ret(self.exprs[:self.size], self.weights[:self.size])
        return ic
        # rank_ic = calculator.calc_pool_rIC_ret(self.exprs[:self.size], self.weights[:self.size])
        # return ic, 
        
    def evaluate_ensemble(self) -> float:
        ic = self.calculator.calc_pool_IC_ret(self.exprs[:self.size], self.weights[:self.size])
        return ic

    @property
    def _under_thres_alpha(self) -> bool:
        if self.ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self.ic_lower_bound

    def calculate_single_ic_for_expr(self, expr: Expression) -> float:
        """计算单个表达式的IC（用于批处理）"""
        try:
            return self.calculator.calc_single_IC_ret(expr)
        except Exception as e:
            print(f"Error calculating IC for expression: {e}")
            return 0.0

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]]]:
        single_ic = self.calculator.calc_single_IC_ret(expr)
        if not self._under_thres_alpha and single_ic < self.ic_lower_bound:
            return single_ic, None

        mutual_ics = []
        for i in range(self.size):
            mutual_ic = self.calculator.calc_mutual_IC(expr, self.exprs[i])
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics

    def _add_factor(
        self,
        expr: Expression,
        ic_ret: float,
        ic_mut: List[float]
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.single_ics[n] = ic_ret
        for i in range(n):
            self.mutual_ics[i][n] = self.mutual_ics[n][i] = ic_mut[i]
        self.weights[n] = ic_ret  # An arbitrary init value
        self.size += 1

    def _pop(self) -> None:
        if self.size <= self.capacity:
            return

        if self.culling_method == 'weight':
            # Original method: remove factor with smallest absolute weight
            worst_idx = np.argmin(np.abs(self.weights[:self.size]))
        elif self.culling_method == 'ic_drop':
            # Remove factor with smallest IC impact (least important)
            current_ic = self.evaluate_ensemble()
            min_ic_drop = float('inf')
            worst_idx = 0

            for i in range(self.size):
                # Temporarily remove factor i and calculate IC
                temp_size = self.size - 1
                if temp_size == 0:
                    continue

                temp_weights = np.delete(self.weights[:self.size], i)
                temp_exprs = [self.exprs[j] for j in range(self.size) if j != i]

                # Normalize weights
                if np.sum(np.abs(temp_weights)) > 0:
                    temp_weights = temp_weights / np.sum(np.abs(temp_weights))

                temp_ic = self.calculator.calc_pool_IC_ret(temp_exprs, temp_weights.tolist())
                ic_drop = current_ic - temp_ic

                if ic_drop < min_ic_drop:
                    min_ic_drop = ic_drop
                    worst_idx = i
        elif self.culling_method == 'combined':
            # Combined method: IC drop * weight importance
            current_ic = self.evaluate_ensemble()
            min_combined_score = float('inf')
            worst_idx = 0

            for i in range(self.size):
                temp_size = self.size - 1
                if temp_size == 0:
                    continue

                temp_weights = np.delete(self.weights[:self.size], i)
                temp_exprs = [self.exprs[j] for j in range(self.size) if j != i]

                if np.sum(np.abs(temp_weights)) > 0:
                    temp_weights = temp_weights / np.sum(np.abs(temp_weights))

                temp_ic = self.calculator.calc_pool_IC_ret(temp_exprs, temp_weights.tolist())
                ic_drop = current_ic - temp_ic
                weight_importance = abs(self.weights[i])

                # Combined score: smaller IC drop + smaller weight = more likely to be removed
                combined_score = ic_drop * (1.0 + weight_importance)

                if combined_score < min_combined_score:
                    min_combined_score = combined_score
                    worst_idx = i
        else:
            raise ValueError(f"Unknown culling method: {self.culling_method}")

        self._swap_idx(worst_idx, self.capacity)
        self.size = self.capacity

    def _swap_idx(self, i, j) -> None:
        if i == j:
            return
        self.exprs[i], self.exprs[j] = self.exprs[j], self.exprs[i]
        self.single_ics[i], self.single_ics[j] = self.single_ics[j], self.single_ics[i]
        self.mutual_ics[:, [i, j]] = self.mutual_ics[:, [j, i]]
        self.mutual_ics[[i, j], :] = self.mutual_ics[[j, i], :]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
        self.single_ics[i], self.single_ics[j] = self.single_ics[j], self.single_ics[i]
        self.mutual_ics[:, [i, j]] = self.mutual_ics[:, [j, i]]
        self.mutual_ics[[i, j], :] = self.mutual_ics[[j, i], :]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
