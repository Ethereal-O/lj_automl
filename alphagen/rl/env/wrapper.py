from typing import Tuple, Optional, Any
import gymnasium as gym
import numpy as np

from alphagen.config import *
from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.rl.env.core import AlphaEnvCore

# 导入字段信息
try:
    from adapters.dic_lol import result_dict as FIELD_DICT
    FIELD_NAMES = list(FIELD_DICT.keys())
except ImportError:
    FIELD_NAMES = ['open', 'close', 'high', 'low', 'volume', 'vwap']

# 扩展动作空间以支持字段模板填充
from alphagen.data.tokens import FIELD_TEMPLATES

# 计算模板填充动作的数量
TEMPLATE_FILL_ACTIONS = 0
for template_name, (base_type, fill_options) in FIELD_TEMPLATES.items():
    TEMPLATE_FILL_ACTIONS += len(fill_options)  # 每个位置的选项数量

SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FIELD_NAMES)  # 使用字段字典中的字段数量
SIZE_TEMPLATE_FILL = TEMPLATE_FILL_ACTIONS  # 模板填充动作
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1

SIZE_ALL = SIZE_NULL + SIZE_OP + SIZE_FEATURE + SIZE_TEMPLATE_FILL + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP
SIZE_ACTION = SIZE_ALL - SIZE_NULL

OFFSET_OP = SIZE_NULL
OFFSET_FEATURE = OFFSET_OP + SIZE_OP
OFFSET_DELTA_TIME = OFFSET_FEATURE + SIZE_FEATURE
OFFSET_CONSTANT = OFFSET_DELTA_TIME + SIZE_DELTA_TIME
OFFSET_SEP = OFFSET_CONSTANT + SIZE_CONSTANT


def action2token(action_raw: int) -> Token:
    action = action_raw + 1
    if action < OFFSET_OP:
        raise ValueError
    elif action < OFFSET_FEATURE:
        return OperatorToken(OPERATORS[action - OFFSET_OP])
    elif action < OFFSET_DELTA_TIME:
        field_name = FIELD_NAMES[action - OFFSET_FEATURE]
        return FeatureToken(field_name)
    elif action < OFFSET_CONSTANT:
        return DeltaTimeToken(DELTA_TIMES[action - OFFSET_DELTA_TIME])
    elif action < OFFSET_SEP:
        return ConstantToken(CONSTANTS[action - OFFSET_CONSTANT])
    elif action == OFFSET_SEP:
        return SequenceIndicatorToken(SequenceIndicatorType.SEP)
    else:
        assert False


class AlphaEnvWrapper(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int
    consecutive_unary_ops: int  # 连续单参数算子计数器
    agent: Optional[Any] = None  # Reference to agent for Q calculation

    def __init__(self, env: AlphaEnvCore):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(SIZE_ACTION)
        self.observation_space = gym.spaces.Box(low=0, high=SIZE_ALL - 1, shape=(MAX_EXPR_LENGTH, ), dtype=np.int32)
        self.consecutive_unary_ops = 0
        self._batch_status = 'running'  # running, waiting_intermediate, waiting_final, terminated

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.int32)
        self.consecutive_unary_ops = 0  # 重置连续单参数算子计数器
        self.env.reset()
        return self.state, {}

    def step(self, action: int):
        # 获取token
        token = self.action(action)

        # 完全简化：不给予任何中间奖励，所有奖励只在最终表达式评估时给出
        # 确保使用CPU设备，避免CUDA问题
        device = torch.device('cpu')
        state_tensor = torch.ByteTensor(self.state).unsqueeze(0).to(device).float()

        try:
            _, reward, done, truncated, info = self.env.step(token, state_tensor)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

        if not done:
            self.state[self.counter] = action
            self.counter += 1

        # 不应用任何额外的penalty或奖励
        final_reward = self.reward(reward)
        return self.state, final_reward, done, truncated, info

    def action(self, action: int) -> Token:
        return action2token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(SIZE_ACTION, dtype=bool)

        # 获取当前表达式状态
        try:
            current_state = self.env._builder.get_expression_state()
            stack_size = len(self.env._builder.stack)
        except AttributeError:
            # 环境还没有reset
            current_state = 'intermediate_invalid'
            stack_size = 0

        # 检查是否处于批处理等待状态
        if hasattr(self, '_batch_status') and self._batch_status == 'waiting':
            # 在批处理等待状态下，不允许任何动作
            return res

        # 获取算子参数数量限制
        max_params = self._get_max_operator_params()

        # 1. 算子选择逻辑 - 语法学习阶段允许所有算子，validator会处理错误
        import os
        is_syntax_learning = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'

        for i in range(OFFSET_OP, OFFSET_OP + SIZE_OP):
            op = OPERATORS[i - OFFSET_OP]
            n_args = op.n_args()

            # 选项逻辑：当前栈大小必须大于等于算子参数数量
            # 这样可以有多个并列部分，也可以选择算子合并
            # 不设并列部分数量的上限限制，只靠episode的step数量限制
            if n_args <= stack_size:
                if is_syntax_learning:
                    # 语法学习阶段：允许所有算子，包括单参数算子
                    # validator会检测irrecoverable状态并惩罚
                    res[i - 1] = True
                else:
                    # IC学习阶段：进行严格的类型检查
                    try:
                        valid_types = self._get_valid_operator_types()
                        if op.category_type() in valid_types:
                            res[i - 1] = True
                    except KeyError:
                        continue

        # 2. 字段选择逻辑 - 给予更高优先级，鼓励构建复杂表达式
        if current_state in ['intermediate_invalid', 'intermediate_valid']:
            for i in range(OFFSET_FEATURE, OFFSET_FEATURE + SIZE_FEATURE):
                field_name = FIELD_NAMES[i - OFFSET_FEATURE]

                # 检查是否是模板字段（需要填充）
                from alphagen.data.tokens import FIELD_TEMPLATES
                if field_name in FIELD_TEMPLATES:
                    # 模板字段：检查是否可以开始填充
                    if self._can_start_field_template(field_name):
                        res[i - 1] = True
                else:
                    # 普通字段：给予更高权重，鼓励扩展表达式
                    # 在中间状态下允许选择字段，构建多字段表达式
                    res[i - 1] = True

        # 3. 常量选择逻辑
        if current_state in ['intermediate_invalid', 'intermediate_valid']:
            for i in range(OFFSET_CONSTANT, OFFSET_CONSTANT + SIZE_CONSTANT):
                if stack_size < max_params:  # 不超过最大参数限制
                    res[i - 1] = True

        # 4. 时间选择逻辑 (暂时禁用)
        # if valid['select'][3]:  # DELTA_TIME
        #     for i in range(OFFSET_DELTA_TIME, OFFSET_DELTA_TIME + SIZE_DELTA_TIME):
        #         res[i - 1] = True

        # 5. SEP选择逻辑 - 语法学习阶段减少单字段停止概率
        import os
        is_syntax_learning = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'

        if current_state == 'irrecoverable':
            # 表达式无法恢复时，允许结束episode（给予负奖励）
            res[OFFSET_SEP - 1] = True
        elif current_state in ['complete', 'intermediate_valid']:
            if is_syntax_learning:
                # 语法学习阶段：降低单字段停止概率，鼓励构建复杂表达式
                try:
                    # 计算当前表达式的字段和算子数量
                    actual_tokens = [token for token in self.env._tokens[1:] if token != SequenceIndicatorToken(SequenceIndicatorToken.SEP)]
                    _, current_features = self.env._parse_expression_components(" ".join(str(t) for t in actual_tokens))

                    if len(current_features) >= 3:
                        # 3个或更多字段：高概率允许结束
                        res[OFFSET_SEP - 1] = True
                    elif len(current_features) >= 2 and len(actual_tokens) >= 3:
                        # 2个字段+至少1个算子：中等概率允许结束
                        res[OFFSET_SEP - 1] = True
                    elif len(current_features) >= 2:
                        # 2个字段但没有算子：低概率允许结束，鼓励添加算子
                        # 只有10%的概率允许结束单字段表达式
                        import random
                        res[OFFSET_SEP - 1] = random.random() < 0.1
                    # 单字段：完全禁止结束，强制继续构建
                except Exception:
                    # 如果解析失败，允许结束避免死锁
                    res[OFFSET_SEP - 1] = True
            else:
                # IC学习阶段：正常允许SEP
                res[OFFSET_SEP - 1] = True

        return res

    def _get_max_operator_params(self) -> int:
        """获取所有算子的最大参数数量"""
        from alphagen.config import OPERATORS
        return max((op.n_args() for op in OPERATORS), default=3)

    def _get_valid_operator_types(self):
        """获取当前允许的算子类型"""
        return {UnaryOperator, BinaryOperator, RollingOperator, PairRollingOperator}

    def _can_start_field_template(self, template_name: str) -> bool:
        """检查是否可以开始填充字段模板"""
        from alphagen.data.tokens import FIELD_TEMPLATES
        if template_name not in FIELD_TEMPLATES:
            return False

        # 检查当前是否已经有未完成的模板
        # 这里可以添加更复杂的逻辑
        return True

    # 批处理支持方法
    def set_batch_status(self, status: str):
        """设置批处理状态"""
        self._batch_status = status

    def get_current_expression(self):
        """获取当前表达式用于批处理计算"""
        try:
            return self.env._builder.get_tree()
        except:
            return None

    def receive_ic_result(self, ic: float):
        """接收IC计算结果（用于批处理）"""
        # 这里可以存储IC结果，用于后续奖励计算
        # 目前简化处理，直接继续
        pass

    def needs_intermediate_ic(self) -> bool:
        """检查是否需要中间IC计算"""
        try:
            current_state = self.env._builder.get_expression_state()
            stack_size = len(self.env._builder.stack)

            # 只有当表达式状态变为intermediate_valid且栈只有一个元素时才需要计算
            import os
            is_syntax_learning = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'

            if is_syntax_learning:
                return False  # 语法学习阶段不计算中间IC

            # 检查是否刚变为intermediate_valid状态
            prev_state = getattr(self, '_prev_state', 'intermediate_invalid')
            current_state = self.env._builder.get_expression_state()

            if (prev_state != 'intermediate_valid' and
                current_state == 'intermediate_valid' and
                len(self.env._builder.stack) == 1):
                self._prev_state = current_state
                return True

            self._prev_state = current_state
            return False

        except AttributeError:
            return False

    def is_at_final_expression(self) -> bool:
        """检查是否到达末尾表达式"""
        try:
            current_state = self.env._builder.get_expression_state()
            return current_state == 'complete'
        except AttributeError:
            return False

    def is_irrecoverable(self) -> bool:
        """检查是否处于无法恢复的状态"""
        try:
            current_state = self.env._builder.get_expression_state()
            return current_state == 'irrecoverable'
        except AttributeError:
            return False


def AlphaEnv(pool: AlphaPoolBase, intermediate_reward_func=None, **kwargs):
    env = AlphaEnvWrapper(AlphaEnvCore(pool=pool, intermediate_reward_func=intermediate_reward_func, **kwargs))
    env.env.sep_action = OFFSET_SEP - 1  # SEP action index
    return env
