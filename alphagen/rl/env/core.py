from typing import Tuple, Optional
import gymnasium as gym
import math
import torch
import numpy as np
from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.utils import reseed_everything
import sys

class AlphaEnvCore(gym.Env):
    pool: AlphaPoolBase
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(self,
                 pool: AlphaPoolBase,
                 device: torch.device = torch.device('cuda:0'),
                 print_expr: bool = False,
                 intermediate_reward_func=None,  # Function to check and compute intermediate reward
                 intermediate_weight=0.2,  # Weight for intermediate rewards
                 final_weight=1.0  # Weight for final reward
                 ):
        super().__init__()

        self.pool = pool
        self._print_expr = print_expr
        self._device = device
        self.intermediate_reward_func = intermediate_reward_func
        self.intermediate_weight = intermediate_weight
        self.final_weight = final_weight
        self.agent = None  # Set later
        self.sep_action = None  # SEP action index

        self.eval_cnt = 0
        self._episode_count = 0  # Initialize episode counter
        self.intermediate_rewards = []  # List of intermediate rewards
        self.last_expression_state = 'intermediate_invalid'  # Track state changes

        # 语法学习阶段跟踪
        self.used_operators = set()  # 已使用过的算子
        self.used_features = set()   # 已使用过的字段

        self.render_mode = None

    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._episode_count += 1
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        self.intermediate_rewards = []  # Reset intermediate rewards
        self.last_expression_state = 'intermediate_invalid'  # Reset state tracking

        # 不重置语法学习跟踪（保持历史记录，鼓励探索新算子和字段）
        # self.used_operators = set()
        # self.used_features = set()

        return self._tokens, self._valid_action_types()

    def step(self, action: Token, state_tensor=None) -> Tuple[List[Token], float, bool, bool, dict]:
        reward = 0.0

        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            # SEP token：完成表达式，计算最终奖励

            # 调试：检查谁允许停止
            sep_allowed_by_mask = getattr(self, '_debug_sep_allowed_by_mask', None)
            if sep_allowed_by_mask is False:
                # action_mask 不允许停止，但仍然停止了，找出原因
                self._debug_who_allowed_stopping("SEP_TOKEN")
            elif sep_allowed_by_mask is True:
                print("🛑 STOP ALLOWED BY: action_mask")
            else:
                print("🛑 STOP ALLOWED BY: unknown (no mask info)")
            final_reward = self._evaluate()
            total_intermediate = sum(self.intermediate_weight * ir for ir in self.intermediate_rewards)
            reward = self.final_weight * final_reward + total_intermediate

            # 更新episode计数器
            episode_num = getattr(self, '_episode_count', 0) + 1
            setattr(self, '_episode_count', episode_num)

            # 检查表达式是否包含至少2个字段
            actual_tokens = [token for token in self._tokens[1:] if token != SequenceIndicatorToken(SequenceIndicatorType.SEP)]
            _, current_features = self._parse_expression_components(" ".join(str(t) for t in actual_tokens))

            # 总是构建token_names（用于文件保存）
            token_names = []
            for token in self._tokens[1:]:  # 跳过BEG_TOKEN
                if isinstance(token, SequenceIndicatorToken):
                    continue
                elif isinstance(token, OperatorToken):
                    if hasattr(token.operator, 'name'):
                        token_names.append(token.operator.name)
                    else:
                        token_names.append(str(token.operator.__name__))
                elif isinstance(token, FeatureToken):
                    token_names.append(f"@{token.feature_name}")
                elif isinstance(token, ConstantToken):
                    token_names.append(str(token.constant))
                else:
                    token_names.append(str(token))

            # 打印所有完整表达式，包括结束原因
            expr_str = ' '.join(token_names)
            end_reason = "NORMAL_END"
            
            # 判断结束原因
            if len(token_names) < 2:
                end_reason = "TOO_SHORT"
            elif "IRRECOVERABLE" in str(reward):
                end_reason = "IRRECOVERABLE"
            elif "INVALID_TOKEN" in str(reward):
                end_reason = "INVALID_TOKEN"
            elif len(self._builder.stack) > 1:
                end_reason = "MULTIPLE_EXPRS"
            elif not self._can_stop_with_single_numeric_element():
                end_reason = "INVALID_TYPE"
            
            # 强制打印所有episode，不管表达式长度
            print(f"Episode {episode_num}: {expr_str} | Reward: {reward:.4f} | {end_reason}")

            # 每100个episode保存一次表达式
            if episode_num % 100 == 0:
                with open(f'expressions_at_episode_{episode_num}.txt', 'w') as f:
                    f.write(f"Episode {episode_num}:\n")
                    f.write(f"Tokens: {' '.join(token_names)}\n")
                    f.write(f"Reward: {reward:.4f}\n")
                    f.write("---\n")
                print(f"Saved expressions at episode {episode_num}")

            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            # 添加token到表达式
            self._tokens.append(action)

            try:
                self._builder.add_token(action)
            except Exception as e:
                # 无效token：强制结束episode并给予负奖励

                # 调试：检查谁允许停止（无效token导致的强制停止）
                sep_allowed_by_mask = getattr(self, '_debug_sep_allowed_by_mask', None)
                action_allowed_by_mask = getattr(self, '_debug_action_allowed_by_mask', None)

                print(f"❌ INVALID TOKEN: {action} was chosen but rejected by validate_op")
                print(f"   Action allowed by mask: {action_allowed_by_mask}")
                print(f"   SEP allowed by mask: {sep_allowed_by_mask}")

                if sep_allowed_by_mask is False:
                    print("🛑 STOP ALLOWED BY: invalid_token (forced termination)")
                elif sep_allowed_by_mask is True:
                    print("🛑 STOP ALLOWED BY: invalid_token (but mask also allowed)")
                else:
                    print("🛑 STOP ALLOWED BY: invalid_token (no mask info)")

                final_reward = self._evaluate_irrecoverable()
                reward = self.final_weight * final_reward + sum(self.intermediate_weight * ir for ir in self.intermediate_rewards)

                # 更新episode计数器
                episode_num = getattr(self, '_episode_count', 0) + 1
                setattr(self, '_episode_count', episode_num)

                # 构建token_names用于显示
                token_names = []
                for token in self._tokens[1:]:
                    if isinstance(token, SequenceIndicatorToken):
                        continue
                    elif isinstance(token, OperatorToken):
                        if hasattr(token.operator, 'name'):
                            token_names.append(token.operator.name)
                        else:
                            token_names.append(str(token.operator.__name__))
                    elif isinstance(token, FeatureToken):
                        token_names.append(f"@{token.feature_name}")
                    elif isinstance(token, ConstantToken):
                        token_names.append(str(token.constant))
                    else:
                        token_names.append(str(token))

                # 不打印INVALID_TOKEN，只记录到文件
                pass

                # 每100个episode保存一次表达式
                if episode_num % 100 == 0:
                    with open(f'expressions_at_episode_{episode_num}.txt', 'w') as f:
                        f.write(f"Episode {episode_num} (INVALID_TOKEN):\n")
                        f.write(f"Tokens: {' '.join(token_names)}\n")
                        f.write(f"Reward: {reward:.4f}\n")
                        f.write("---\n")
                    print(f"Saved invalid expressions at episode {episode_num}")

                done = True
                return self._tokens, reward, done, False, self._valid_action_types()

            # 移除irrecoverable状态检查 - action_mask确保不会出现无效状态
            # if self._builder.get_expression_state() == 'irrecoverable':
            #     ...

            done = False

            # IC学习阶段：正常计算中间IC奖励
            prev_state = self._builder.get_expression_state()
            self._builder.get_expression_state()  # 更新状态

            if prev_state != 'intermediate_valid' and self._builder.get_expression_state() == 'intermediate_valid':
                # 状态变为合理的中间表达式，计算IC奖励
                try:
                    if len(self._builder.stack) == 1:
                        expr: Expression = self._builder.get_tree()

                        # Calculate IC increment: new_pool_IC - old_pool_IC
                        old_pool_ic = self.pool.evaluate_ensemble() if self.pool.size > 0 else 0.0

                        # Temporarily add expression to pool and calculate new IC
                        temp_ic_ret, temp_ic_mut = self.pool._calc_ics(expr)
                        if temp_ic_ret is not None and temp_ic_mut is not None:
                            old_size = self.pool.size
                            self.pool._add_factor(expr, temp_ic_ret, temp_ic_mut)

                            # Calculate new pool IC with optimization
                            if self.pool.size > 1:
                                new_weights = self.pool._optimize(alpha=self.pool.l1_alpha, lr=5e-4, n_iter=100)
                                self.pool.weights[:self.pool.size] = new_weights

                            new_pool_ic = self.pool.evaluate_ensemble()

                            # Restore pool state
                            self.pool.size = old_size
                            self.pool.exprs[old_size] = None
                            self.single_ics[old_size] = 0.0
                            self.weights[old_size] = 0.0
                            if old_size > 0:
                                self.mutual_ics[old_size, :old_size] = 0.0
                                self.mutual_ics[:old_size, old_size] = 0.0

                            # Calculate IC increment as reward
                            ic_increment = new_pool_ic - old_pool_ic
                            ic_reward = self.intermediate_weight * ic_increment

                            reward = ic_reward
                            self.intermediate_rewards.append(ic_increment)
                        else:
                            reward = -0.01
                    else:
                        reward = -0.01
                except Exception as e:
                    # 只在关键错误时打印
                    if "OutOfDataRange" not in str(e):
                        print(f"Intermediate IC calculation error: {e}", file=sys.stderr)
                    reward = -0.01
            elif self._builder.get_expression_state() == 'irrecoverable':
                reward = -0.1
            else:
                reward = -0.001
        else:
            # 超出最大长度，强制结束

            # 调试：检查谁允许停止（超出最大长度导致的强制停止）
            sep_allowed_by_mask = getattr(self, '_debug_sep_allowed_by_mask', None)
            if sep_allowed_by_mask is False:
                print("🛑 STOP ALLOWED BY: max_length_exceeded (forced termination)")
            elif sep_allowed_by_mask is True:
                print("🛑 STOP ALLOWED BY: max_length_exceeded (but mask also allowed)")
            else:
                print("🛑 STOP ALLOWED BY: max_length_exceeded (no mask info)")

            done = True
            final_reward = self._evaluate() if self._builder.is_valid() else -1.
            total_intermediate = sum(self.intermediate_weight * ir for ir in self.intermediate_rewards)
            reward = self.final_weight * final_reward + total_intermediate

        if math.isnan(reward):
            reward = 0.

        truncated = False
        return self._tokens, reward, done, truncated, self._valid_action_types()

    def _evaluate_irrecoverable(self) -> float:
        """
        评估irrecoverable表达式：给予负奖励
        """
        import os
        is_syntax_learning_phase = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'

        if is_syntax_learning_phase:
            # 语法学习阶段：irrecoverable表达式给予较大负奖励
            return -1.0
        else:
            # IC学习阶段：irrecoverable表达式给予惩罚
            return -0.5

    def _evaluate(self):
        # 检查是否处于语法学习阶段（通过环境变量控制）
        import os
        is_syntax_learning_phase = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'

        # 检查栈状态
        stack_size = len(self._builder.stack)

        if stack_size == 0:
            # 空栈：完全没有表达式
            if self._print_expr:
                print("Warning: Empty expression stack - no expression to evaluate")
            return -1.0

        elif stack_size > 1:
            # 多个表达式：检查是否是语法学习阶段
            if is_syntax_learning_phase:
                # 语法学习阶段：奖励并列表达式
                try:
                    syntax_reward = self._calculate_syntax_reward()
                    if self._print_expr:
                        print(f"Syntax learning: {stack_size} parallel expressions, reward: {syntax_reward:.4f}")
                    return syntax_reward
                except Exception as e:
                    if self._print_expr:
                        print(f"Syntax reward calculation failed: {e}")
                    return -0.5
            else:
                # IC学习阶段：惩罚未完成的表达式
                if self._print_expr:
                    print(f"Warning: Incomplete expression - {stack_size} expressions in stack, should not stop here")
                    for i, expr in enumerate(self._builder.stack):
                        print(f"  Expression {i}: {expr}")
                return -2.0

        else:  # stack_size == 1
            # 单个表达式：检查学习阶段
            if is_syntax_learning_phase:
                # 语法学习阶段：计算语法奖励（不计算真实IC）
                try:
                    expr: Expression = self._builder.get_tree()
                    expr_str = str(expr)
                    syntax_reward = self._calculate_syntax_reward_only(expr_str)
                    return syntax_reward
                except Exception as e:
                    if self._print_expr:
                        print(f"Syntax reward calculation failed: {e}")
                    return -0.5
            else:
                # IC学习阶段：正常计算IC
                try:
                    expr: Expression = self._builder.get_tree()

                    # 检查是否处于预热阶段（避免在memory未满时计算IC）
                    if hasattr(self.pool.calculator, '_agent_ref') and self.pool.calculator._agent_ref():
                        agent = self.pool.calculator._agent_ref()
                        if hasattr(agent, 'memory') and agent.memory.size() < 10000:
                            # 预热阶段：返回0奖励，不计算真实IC
                            return 0.0

                    # 计算因子奖励
                    ret = self.pool.try_new_expr(expr)
                    self.eval_cnt += 1
                    return ret
                except OutOfDataRangeError:
                    return 0.
                except Exception as e:
                    if self._print_expr:
                        print(f"Expression evaluation failed: {e}")
                    return -1.0

    def _can_stop_with_single_numeric_element(self) -> bool:
        """
        检查是否可以停止：栈中有且仅有一个完整元素且为数值类型时才允许停止
        """
        try:
            stack = self._builder.stack

            # 必须有且仅有一个并列部分（所有算子都应用完毕）
            if len(stack) != 1:
                return False

            single_part = stack[0]

            # 这个部分必须是完整的（featured）
            if not getattr(single_part, 'is_featured', False):
                return False

            # 这个部分的类型必须是数值类型（float或int）
            # 从表达式字符串推断类型
            expr_str = str(single_part)

            # 解析表达式并检查类型
            try:
                from adapters.expression_validator import validate_expression_types
                valid, actual_type, _ = validate_expression_types(expr_str)
                if not valid:
                    return False

                # 检查是否是数值类型
                if actual_type not in ['float', 'int', 'const_float', 'const_int']:
                    return False

                return True

            except Exception:
                # 类型检查失败时保守返回False
                return False

        except Exception:
            # 如果检查失败，保守地返回False
            return False

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_feature()

        # 语法学习阶段：SEP可用性由wrapper中的概率逻辑决定
        # 这里只提供基本的状态信息
        current_state = self._builder.get_expression_state()
        has_single_tree = len(self._builder.stack) == 1

        # 停止条件：栈中有且仅有一个完整元素且为数值类型时才允许停止
        valid_stop = self._can_stop_with_single_numeric_element()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret

    def _rebuild_expression_from_tokens(self, tokens: List[Token]) -> str:
        """
        从token序列重建完整的表达式字符串
        将逆波兰表达式的token转换为正常的中缀表达式
        """
        if not tokens:
            return ""

        stack = []

        for token in tokens:
            if isinstance(token, OperatorToken):
                # 算子：从栈中弹出相应数量的操作数
                n_args = token.operator.n_args()
                if len(stack) < n_args:
                    # 参数不足，返回部分结果
                    args = stack + ["?"] * (n_args - len(stack))
                else:
                    args = [stack.pop() for _ in range(n_args)][::-1]  # 逆波兰式参数顺序

                # 构建算子调用
                op_name = str(token.operator)
                if n_args == 1:
                    expr = f"{op_name}({args[0]})"
                elif n_args == 2:
                    expr = f"{op_name}({args[0]}, {args[1]})"
                elif n_args == 3:
                    expr = f"{op_name}({args[0]}, {args[1]}, {args[2]})"
                else:
                    # 对于更多参数，用逗号分隔
                    args_str = ", ".join(str(arg) for arg in args)
                    expr = f"{op_name}({args_str})"

                stack.append(expr)

            elif isinstance(token, FeatureToken):
                # 字段：直接压栈
                if hasattr(token, 'feature_name'):
                    stack.append(f"@{token.feature_name}")
                else:
                    stack.append(str(token.feature))

            elif isinstance(token, ConstantToken):
                # 常量：直接压栈
                stack.append(str(token.constant))

            elif isinstance(token, DeltaTimeToken):
                # 时间差：直接压栈
                stack.append(str(token.delta_time))

            else:
                # 其他token：转为字符串
                stack.append(str(token))

        # 如果栈中有多个元素，返回逗号分隔的并列表达式
        if len(stack) > 1:
            return ", ".join(stack)
        elif len(stack) == 1:
            return stack[0]
        else:
            return ""

    def _calculate_syntax_reward(self) -> float:
        """
        计算语法学习阶段的奖励
        奖励：合法性、新颖性、多样性、复杂度
        """
        try:
            # 重建当前表达式的字符串
            actual_tokens = [token for token in self._tokens[1:] if token != SequenceIndicatorToken(SequenceIndicatorType.SEP)]
            expr_str = self._rebuild_expression_from_tokens(actual_tokens)

            if not expr_str:
                return -0.5  # 空表达式

            # 解析表达式中的组件
            current_ops, current_features = self._parse_expression_components(expr_str)

            # 1. 合法性奖励：基础分数
            validity_reward = 1.0

            # 2. 新颖性奖励：新算子和字段
            novelty_reward = 0.0

            # 新算子奖励
            new_ops = current_ops - self.used_operators
            novelty_reward += len(new_ops) * 0.5

            # 新字段奖励
            new_features = current_features - self.used_features
            novelty_reward += len(new_features) * 0.3

            # 更新已使用集合
            self.used_operators.update(current_ops)
            self.used_features.update(current_features)

            # 3. 多样性奖励：算子类型多样性
            diversity_reward = self._calculate_diversity_reward(current_ops)

            # 4. 复杂度奖励：基于表达式长度和算子数量
            complexity_reward = min(len(actual_tokens) * 0.01, 0.5)  # 最大0.5

            # 5. 并列表达式奖励：多个并列部分额外奖励
            stack_size = len(self._builder.stack)
            parallel_reward = max(0, (stack_size - 1) * 0.1)  # 每个额外并列部分+0.1

            # 综合奖励
            total_reward = (
                1.0 * validity_reward +      # 1.0 * 1.0
                1.0 * novelty_reward +       # 新颖性权重
                0.5 * diversity_reward +     # 多样性权重
                0.3 * complexity_reward +    # 复杂度权重
                0.2 * parallel_reward        # 并列奖励权重
            )

            if self._print_expr:
                print(f"  Syntax components: ops={current_ops}, features={current_features}")
                print(f"  Syntax rewards: validity={validity_reward:.2f}, novelty={novelty_reward:.2f}, "
                      f"diversity={diversity_reward:.2f}, complexity={complexity_reward:.2f}, "
                      f"parallel={parallel_reward:.2f}")

            return total_reward

        except Exception as e:
            if self._print_expr:
                print(f"Syntax reward calculation error: {e}")
            return -0.5

    def _parse_expression_components(self, expr_str: str) -> Tuple[set, set]:
        """
        解析表达式字符串，提取算子和字段
        """
        operators = set()
        features = set()

        # 简单解析：查找算子和字段模式
        import re

        # 算子模式：单词后面跟括号
        op_pattern = r'\b([A-Za-z_]\w*)\s*\('
        ops_found = re.findall(op_pattern, expr_str)
        operators.update(ops_found)

        # 字段模式：@后面跟字段名
        field_pattern = r'@([A-Za-z_]\w*)'
        fields_found = re.findall(field_pattern, expr_str)
        features.update(fields_found)

        return operators, features

    def _calculate_diversity_reward(self, operators: set) -> float:
        """
        计算算子类型的多样性奖励
        """
        if not operators:
            return 0.0

        # 算子类型分类
        type_counts = {
            'vec': 0,    # 向量算子
            'cs': 0,     # 截面算子
            'ts': 0,     # 时序算子
            'basic': 0   # 基础算子
        }

        for op in operators:
            op_str = op.lower()
            if op_str.startswith('vec') or 'vector' in op_str:
                type_counts['vec'] += 1
            elif op_str.startswith('cs'):
                type_counts['cs'] += 1
            elif op_str.startswith('ts') or 'time' in op_str:
                type_counts['ts'] += 1
            else:
                type_counts['basic'] += 1

        # 计算使用的类型数量
        used_types = sum(1 for count in type_counts.values() if count > 0)

        # 多样性奖励：每多使用一种类型+0.2
        diversity_reward = used_types * 0.2

        return diversity_reward

    def _calculate_syntax_reward_only(self, expr_str: str) -> float:
        """
        只计算语法奖励，不进行真实的因子计算
        用于语法学习阶段，避免调用pool.try_new_expr
        """
        try:
            if not expr_str:
                return -0.5  # 空表达式

            # 解析表达式中的组件
            current_ops, current_features = self._parse_expression_components(expr_str)

            # 1. 合法性奖励：语法学习阶段不给合法性奖励，避免过早完成简单表达式
            import os
            is_syntax_learning = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'

            if is_syntax_learning:
                # 语法学习阶段：不给合法性奖励，鼓励探索复杂表达式
                validity_reward = 0.0
            else:
                # IC学习阶段：正常合法性奖励
                validity_reward = self._check_expression_validity(expr_str, current_ops, current_features)

            # 2. 新颖性奖励：新算子和字段
            novelty_reward = 0.0

            # 新算子奖励
            new_ops = current_ops - self.used_operators
            novelty_reward += len(new_ops) * 0.4

            # 新字段奖励
            new_features = current_features - self.used_features
            novelty_reward += len(new_features) * 0.3

            # 更新已使用集合
            self.used_operators.update(current_ops)
            self.used_features.update(current_features)

            # 3. 多样性奖励：算子类型多样性
            diversity_reward = self._calculate_diversity_reward(current_ops)

            # 4. 复杂度奖励：基于多参数算子和字段组合
            complexity_reward = self._calculate_expression_complexity(current_ops, current_features)

            # 5. 字段组合奖励：大幅鼓励多个字段之间的运算
            combination_reward = 0.0
            if len(current_features) >= 2:
                combination_reward = min(len(current_features) - 1, 3) * 2.0  # 大幅提高权重

            # 6. 惩罚单参数算子堆叠：大幅惩罚单参数算子
            unary_penalty = 0.0
            unary_ops = self._count_unary_operators(expr_str)
            if unary_ops > 0:  # 只要有单参数算子就开始惩罚
                unary_penalty = unary_ops * 0.3  # 每个单参数算子扣0.3分

            # 7. 奖励多参数算子：如果有双参数或多参数算子，给额外奖励
            multi_param_bonus = 0.0
            from alphagen.config import OPERATORS
            for op in OPERATORS:
                if hasattr(op, 'name') and op.name in current_ops:
                    if op.n_args() >= 2:
                        multi_param_bonus += 0.4  # 每个多参数算子加0.4分

            # 语法学习阶段：只有真正复杂的表达式才给奖励
            if is_syntax_learning:
                # 语法学习：只奖励使用多个字段且多个算子的复杂表达式
                if len(current_features) >= 2 and len(current_ops) >= 2:
                    total_reward = (
                        1.5 * novelty_reward +          # 新颖性奖励
                        1.0 * diversity_reward +        # 多样性奖励
                        2.0 * combination_reward +      # 多字段组合奖励
                        1.5 * multi_param_bonus         # 多参数算子奖励
                    )
                else:
                    total_reward = -0.5  # 简单表达式不给奖励
            else:
                # IC学习阶段：正常奖励
                total_reward = (
                    1.0 * validity_reward +
                    1.5 * novelty_reward +
                    1.0 * diversity_reward +
                    1.2 * complexity_reward +
                    1.5 * combination_reward +
                    1.0 * multi_param_bonus -
                    0.5 * unary_penalty
                )

            return total_reward

        except Exception as e:
            return -0.5

    def _calculate_expression_complexity(self, ops: set, features: set) -> float:
        """
        计算表达式复杂度奖励
        鼓励多参数算子和字段组合，不鼓励单参数算子堆叠
        """
        if not ops and not features:
            return 0.0

        # 计算多参数算子数量
        multi_param_ops = 0
        from alphagen.config import OPERATORS
        for op in OPERATORS:
            if hasattr(op, 'name') and op.name in ops:
                if op.n_args() >= 2:  # 双参数或多参数算子
                    multi_param_ops += 1
                elif op.n_args() == 1 and len(features) >= 2:  # 单参数算子但有多个字段
                    multi_param_ops += 0.5  # 给一半奖励

        # 复杂度奖励
        complexity_score = min(multi_param_ops * 0.3 + len(features) * 0.1, 1.0)

        return complexity_score

    def _count_unary_operators(self, expr_str: str) -> int:
        """
        统计表达式中的单参数算子数量
        """
        try:
            from alphagen.config import OPERATORS
            tokens = expr_str.split()

            unary_count = 0
            for token in tokens:
                # 查找算子
                for op in OPERATORS:
                    if hasattr(op, 'name') and op.name == token:
                        if op.n_args() == 1:
                            unary_count += 1
                        break

            return unary_count

        except Exception:
            return 0

    def _check_expression_validity(self, expr_str: str, ops: set, features: set) -> float:
        """
        检查表达式的合法性
        返回0-1之间的分数：1.0表示完全合法，0.0表示无效
        """
        try:
            from adapters.废弃算子规则 import get_type_compatibility, OPERATOR_SIGNATURES
            from adapters.dic_lol import result_dict as FIELD_DICT

            # 基本检查
            if not expr_str or not features:
                return 0.0  # 必须有至少一个字段

            # 解析token序列
            tokens = expr_str.split()
            if not tokens:
                return 0.0

            # 模拟逆波兰表达式求值，检查类型匹配
            type_stack = []

            for token in tokens:
                if token.startswith('@'):
                    # 字段：从字段字典获取类型
                    field_name = token[1:]  # 去掉@
                    if field_name in FIELD_DICT:
                        field_type = FIELD_DICT[field_name][0]  # (type, choices)
                        type_stack.append(field_type)
                    else:
                        return 0.0  # 未知字段
                elif token.replace('.', '').replace('-', '').isdigit() or token in ['0.0', '1.0', '2.5', '3.0', '5.0', '10.0', '15.5', '20.0', '30.0', '50.0', '100.0', '120.0', '240.0', '505.0', '1000.0', '100000500.0', '200000000.0', '-3.0']:
                    # 常量：根据值推断类型
                    if '.' in token or 'e' in token.lower():
                        type_stack.append('const_float')
                    else:
                        type_stack.append('const_int')
                else:
                    # 算子：检查参数类型
                    if token in OPERATOR_SIGNATURES:
                        arg_types, return_type = OPERATOR_SIGNATURES[token]
                        n_args = len(arg_types)

                        if len(type_stack) < n_args:
                            return 0.0  # 参数不足

                        # 检查参数类型匹配
                        args_on_stack = type_stack[-n_args:]
                        total_compatibility = 0.0

                        for i, expected_type in enumerate(arg_types):
                            actual_type = args_on_stack[i]
                            compatibility = get_type_compatibility(expected_type, actual_type)
                            total_compatibility += compatibility

                        # 平均兼容性必须 > 0.5
                        avg_compatibility = total_compatibility / n_args
                        if avg_compatibility <= 0.5:
                            return 0.0  # 类型不兼容

                        # 弹出参数，压入返回值
                        for _ in range(n_args):
                            type_stack.pop()
                        type_stack.append(return_type)
                    else:
                        return 0.0  # 未知算子

            # 检查最终结果
            if len(type_stack) != 1:
                return 0.0  # 应该只剩一个结果

            final_type = type_stack[0]
            # 最终结果必须是数值类型
            if final_type not in ['float', 'int', 'const_float', 'const_int']:
                return 0.0  # 必须产生数值

            # 检查算子使用合理性
            if len(ops) > 20:
                return 0.1  # 算子太多，降低分数

            if len(ops) == 0 and len(features) == 1:
                return 0.8  # 简单字段，高分但不是满分

            if len(features) > len(ops) + 2:
                return 0.2  # 太多字段，算子太少

            # 奖励合理复杂度的表达式
            complexity_score = min(len(tokens) / 10.0, 1.0)  # 基于长度
            return min(1.0, complexity_score)

        except Exception:
            return 0.0

    def _debug_who_allowed_stopping(self, stop_reason: str):
        """
        调试函数：分析为什么停止被允许了
        当 action_mask 不允许停止但仍然停止时调用
        """
        print(f"🔍 DEBUG: Investigating why stop was allowed despite mask forbidding it")
        print(f"   Stop reason: {stop_reason}")
        print(f"   Stack size: {len(self._builder.stack)}")

        # 检查栈状态
        if len(self._builder.stack) == 0:
            print("   Stack is empty - this should not happen")
            return

        # 检查每个栈元素的类型
        for i, expr in enumerate(self._builder.stack):
            expr_type = self._infer_expr_type(expr)
            print(f"   Stack[{i}]: {expr} -> type: {expr_type}")

        # 检查是否符合停止条件
        if len(self._builder.stack) == 1:
            single_expr = self._builder.stack[0]
            expr_type = self._infer_expr_type(single_expr)

            if expr_type in ['float', 'int']:
                print(f"   ✅ Single numeric expression: type={expr_type}")
                print("   This should have been allowed by action_mask!")
            else:
                print(f"   ❌ Single expression but wrong type: {expr_type} (expected float/int)")
        else:
            print(f"   ❌ Multiple expressions in stack: {len(self._builder.stack)}")

        # 检查语法学习阶段
        import os
        is_syntax_learning = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'
        print(f"   Syntax learning phase: {is_syntax_learning}")

        # 检查 _can_stop_with_single_numeric_element
        can_stop = self._can_stop_with_single_numeric_element()
        print(f"   _can_stop_with_single_numeric_element(): {can_stop}")

    def _infer_expr_type(self, expr) -> str:
        """
        简单推断表达式类型（用于调试）
        """
        try:
            # 类似 wrapper 中的 _infer_type 方法
            if hasattr(expr, 'feature'):
                feat_name = str(expr.feature).replace("Feature.", "").replace("@", "").strip("'\"")
                from alphagen.rl.env.wrapper import AlphaEnvWrapper
                # 这里需要访问 wrapper 的类型映射，但简化处理
                return "float"  # 默认

            if hasattr(expr, '_value'):
                return "int" if isinstance(expr._value, int) else "float"

            # 检查算子类型
            op_name = getattr(expr, 'name', expr.__class__.__name__)
            from adapters.operator_library import OPERATOR_SIGNATURES
            if op_name in OPERATOR_SIGNATURES:
                _, return_type = OPERATOR_SIGNATURES[op_name]
                return return_type

            return "unknown"
        except:
            return "error"

    def _validate_factor_computation(self, expr: Expression, ic_value: float):
        """
        验证因子计算结果，确保因子值真的被计算出来了
        如果验证失败，立即停止程序
        """
        import sys  # 确保在函数内部也能访问 sys 模块
        expr_str = str(expr)

        # 检查IC值是否有效
        if ic_value is None or (isinstance(ic_value, float) and math.isnan(ic_value)):
            print(f"❌ FATAL ERROR: Factor IC calculation returned NaN for expression: {expr_str}", file=sys.stderr)
            print("Program will terminate immediately.", file=sys.stderr)
            sys.exit(1)

        # 检查因子值是否被缓存（表示计算成功）
        from adapters.scoring_calculator import factor_cache
        if not factor_cache.has_factor(expr_str):
            print(f"❌ FATAL ERROR: Factor values were not computed and cached for expression: {expr_str}", file=sys.stderr)
            print("This indicates the external computation engine failed to produce valid factor data.", file=sys.stderr)
            print("Program will terminate immediately.", file=sys.stderr)
            sys.exit(1)

        # 验证缓存的数据质量
        try:
            cached_data = factor_cache.load_factor(expr_str)
            values = cached_data['values']
            dates = cached_data['dates']
            symbols = cached_data['symbols']

            # 检查数据维度
            if values is None or len(values.shape) != 2:
                print(f"❌ FATAL ERROR: Invalid factor data shape for expression: {expr_str}", file=sys.stderr)
                print(f"Expected 2D array, got: {values.shape if values is not None else None}", file=sys.stderr)
                sys.exit(1)

            # 检查是否有足够的日期和股票数据
            n_dates, n_stocks = values.shape
            if n_dates < 10 or n_stocks < 10:
                print(f"❌ FATAL ERROR: Insufficient data dimensions for expression: {expr_str}", file=sys.stderr)
                print(f"Dates: {n_dates}, Stocks: {n_stocks}", file=sys.stderr)
                sys.exit(1)

            # 检查数据是否全为NaN或零
            if np.all(np.isnan(values)) or np.all(values == 0):
                print(f"❌ FATAL ERROR: Factor data is all NaN or zero for expression: {expr_str}", file=sys.stderr)
                print("This indicates the computation engine produced invalid results.", file=sys.stderr)
                sys.exit(1)

        except Exception as e:
            print(f"❌ FATAL ERROR: Failed to validate cached factor data for expression: {expr_str}", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode='human'):
        pass
