from typing import Tuple, Optional, Any, List, Dict
import gymnasium as gym
import numpy as np
import torch

from alphagen.config import *
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.rl.env.core import AlphaEnvCore

# ==========================================
# 1. 外部依赖与配置加载
# ==========================================
try:
    from adapters.dic_lol import result_dict
    from adapters.field_config import field_config
    FIELD_NAMES = field_config.get_field_names()
    from adapters.operator_library import OPERATOR_SIGNATURES
except ImportError:
    result_dict = {}
    FIELD_NAMES = []
    OPERATOR_SIGNATURES = {}

# 动作空间偏移量定义
SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FIELD_NAMES)
SIZE_SEP = 1
SIZE_ALL = SIZE_NULL + SIZE_OP + SIZE_FEATURE + SIZE_SEP
SIZE_ACTION = SIZE_ALL - SIZE_NULL

OFFSET_OP = SIZE_NULL
OFFSET_FEATURE = OFFSET_OP + SIZE_OP
OFFSET_SEP = OFFSET_FEATURE + SIZE_FEATURE # SEP 紧跟在特征之后
MAX_SEQ_LENGTH = 256

def action2token(action_raw: int) -> Token:
    """将 Agent 选出的整数动作索引转为 Token 对象"""
    action = action_raw + 1
    if action < OFFSET_FEATURE:
        return OperatorToken(OPERATORS[action - OFFSET_OP])
    elif action < OFFSET_SEP:
        return FeatureToken(FIELD_NAMES[action - OFFSET_FEATURE])
    elif action == OFFSET_SEP:
        return SequenceIndicatorToken(SequenceIndicatorType.SEP)
    raise ValueError(f"Action index {action_raw} is invalid.")

class AlphaEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(SIZE_ACTION)
        self.observation_space = gym.spaces.Box(
            low=0, high=SIZE_ALL, shape=(MAX_SEQ_LENGTH,), dtype=np.int32
        )
        # 预缓存类型映射，加速 Mask 计算
        self._feature_type_map = {k: v[0] for k, v in result_dict.items()}

    # ==========================================
    # 2. 核心数据转换逻辑 (解决 TypeError)
    # ==========================================

    def _get_token_id(self, token: Token) -> int:
        """Token 对象 -> 整数 ID"""
        if isinstance(token, SequenceIndicatorToken):
            if token.indicator == SequenceIndicatorType.BEG: return 0
            if token.indicator == SequenceIndicatorType.SEP: return OFFSET_SEP
        elif isinstance(token, OperatorToken):
            op_name = str(token)
            for i, op in enumerate(OPERATORS):
                curr_name = getattr(op, 'name', op.__name__ if hasattr(op, '__name__') else str(op))
                if curr_name == op_name: return OFFSET_OP + i
        elif isinstance(token, FeatureToken):
            try: return OFFSET_FEATURE + FIELD_NAMES.index(token.feature_name)
            except ValueError: return 0
        return 0

    def _pad_obs(self, tokens: List[Token]) -> np.ndarray:
        """核心修复：将底层 Token 列表转为 numpy int32 数组"""
        token_ids = [self._get_token_id(t) for t in tokens]
        token_ids = token_ids[:MAX_SEQ_LENGTH]
        # 使用常量 0 (NULL) 进行填充
        return np.pad(token_ids, (0, MAX_SEQ_LENGTH - len(token_ids)), 'constant', constant_values=0).astype(np.int32)

    # ==========================================
    # 3. 重写 reset 和 step (拦截并转化数据流)
    # ==========================================

    def reset(self, **kwargs):
        obs_raw, info = self.env.reset(**kwargs)
        return self._pad_obs(obs_raw), info

    def step(self, action: int):
        # 记录当前的action_mask状态，用于调试谁允许停止
        current_mask = self.action_masks()
        sep_allowed_by_mask = current_mask[OFFSET_SEP - 1] if OFFSET_SEP - 1 < len(current_mask) else False

        # 检查动作是否被mask允许
        action_allowed_by_mask = current_mask[action] if action < len(current_mask) else False

        # 存储到环境对象中，供core使用
        self.env._debug_sep_allowed_by_mask = sep_allowed_by_mask
        self.env._debug_action_allowed_by_mask = action_allowed_by_mask

        # 1. 将整数 Action 转换为底层 Core 需要的 Token 对象
        token = action2token(action)

        # 2. 🚩 关键修改：传给内层环境的是 token 对象，而不是 action 整数
        obs_raw, reward, terminated, truncated, info = self.env.step(token)

        # 3. 转化观测值
        obs = self._pad_obs(obs_raw)

        done = terminated or truncated
        if done:
            self._print_episode_summary(action, token, reward, terminated, truncated, info)

        return obs, reward, terminated, truncated, info

    # ==========================================
    # 4. 类型检查与 Action Mask (解决 Incomplete)
    # ==========================================

    def _infer_type(self, expr: Any) -> str:
        """递归推断表达式类型，不依赖 expr.return_type"""
        # 1. 如果是 Feature (叶子节点)
        if hasattr(expr, 'feature'):
            # 处理 FeatureToken 或 FeatureExpression
            feat_name = str(expr.feature).replace("Feature.", "").replace("@", "").strip("'\"")
            return self._feature_type_map.get(feat_name, "float")  # 默认float

        # 2. 如果是 Constant (常量)
        if hasattr(expr, '_value'):
            if isinstance(expr._value, int):
                return "int"
            return "float"

        # 3. 如果是 Operator (算子节点)
        op_name = getattr(expr, 'name', expr.__class__.__name__)
        if op_name in OPERATOR_SIGNATURES:
            _, return_type = OPERATOR_SIGNATURES[op_name]
            return return_type

        # 4. 未知类型，保守返回float
        return 'float'

    def _is_subtype(self, actual: str, expected: str) -> bool:
        """判定类型兼容性"""
        if expected in ["Any", "expr"]: return True
        if actual == expected: return True
        if expected == "vector" and "vector" in actual: return True
        if expected == "float" and actual == "int": return True
        return False

    def action_masks(self) -> np.ndarray:
        # 步骤 A: 初始化掩码
        mask = np.zeros(self.action_space.n, dtype=bool)
        stack = self.env._builder.stack

        # 步骤 B: 堆栈为空时的处理 (Start State)
        if len(stack) == 0:
            # 仅允许选择特征（Features）
            mask[OFFSET_FEATURE - 1 : OFFSET_SEP - 1] = True
            # 禁止所有算子（Operators）和停止符（SEP）
            return mask

        # 步骤 C: 堆栈非空时的类型推断 (Type Inference)
        top_type = self._infer_type(stack[-1])

        # 步骤 D: 遍历算子生成掩码 (Operator Masking)
        for i, op in enumerate(OPERATORS):
            op_name = getattr(op, 'name', op.__name__ if hasattr(op, '__name__') else str(op))
            sig = OPERATOR_SIGNATURES.get(op_name)
            if not sig:
                continue
            arg_types, _ = sig
            n_args = len(arg_types)
            if n_args == 0:
                # 零参数算子（如 IsToday）通常不接在表达式后面
                continue
            if len(stack) < n_args:
                continue
            # RPN 逻辑：检查栈顶 n_args 个元素是否匹配算子的所有参数要求
            match = True
            for j in range(n_args):
                req_type = arg_types[j]  # 第 j 个参数的需求类型
                actual_type = self._infer_type(stack[-(n_args - j)])  # 对应的栈元素类型
                if not (req_type in ['any', 'expr'] or
                        req_type == actual_type or
                        (req_type == 'float' and actual_type == 'int')):
                    # 严禁将 vector 传给只接受 float/int 的算子位置
                    if req_type in ['float', 'int'] and actual_type == 'vector':
                        match = False
                        break
                    match = False
                    break
            if match:
                mask[OFFSET_OP + i - 1] = True

        # 步骤 E: 特征动作 (Feature Masking)
        # 在堆栈非空时，始终允许压入新的特征（开启新分支）
        mask[OFFSET_FEATURE - 1 : OFFSET_SEP - 1] = True

        # 步骤 F: 停止条件 (SEP Masking)
        if len(stack) == 1 and top_type in ['float', 'int']:
            mask[OFFSET_SEP - 1] = True

        return mask

    def valid_action_mask(self) -> np.ndarray:
        return self.action_masks()

    # ==========================================
    # 5. 结算打印逻辑
    # ==========================================

    def _print_episode_summary(self, action, token, reward, terminated, truncated, info):
        token_history = self.env._tokens
        action_sequence = " ".join([str(t) for t in token_history])

        builder = self.env._builder
        try:
            if len(builder.stack) == 1:
                expr_str = str(builder.get_tree())
            else:
                expr_str = "[Incomplete] " + " | ".join([str(e) for e in builder.stack])
        except:
            expr_str = "Parse Error"

        # 确定停止原因
        if truncated:
            reason = "Timeout (Max Length)"
        elif action == OFFSET_SEP - 1:
            reason = "Agent Manual Stop (SEP)"
        else:
            reason = info.get("error", "Core Terminated")
        print("\n=== Episode Summary ===")
        print(f"Total expr: {expr_str} | Terminated: {terminated} | Truncated: {truncated} | Reason: {reason}")


# ==========================================
# 6. 环境构造工厂
# ==========================================
def AlphaEnv(pool: AlphaPoolBase, intermediate_reward_func=None, **kwargs):
    core = AlphaEnvCore(pool=pool, intermediate_reward_func=intermediate_reward_func, **kwargs)
    return AlphaEnvWrapper(core)
