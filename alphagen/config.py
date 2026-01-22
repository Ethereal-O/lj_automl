import os
import sys
import torch

# 1. 环境路径配置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 基础配置
MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

# 3. 导入外部定义的签名库和字段字典
from adapters.operator_library import OPERATOR_SIGNATURES
from adapters.dic_lol import result_dict
from adapters.field_config import field_config
FIELD_NAMES = field_config.get_field_names()

from alphagen.data.expression import Operator
from alphagen_qlib.stock_data import StockData

# ====================================================
# 4. 通用算子类：仅提供 RPN 构建所需的元数据
# ====================================================

class CustomOperator(Operator):
    def __init__(self, name, arg_types, return_type):
        self.name = name
        self.arg_types = arg_types
        self.return_type = return_type
        self._n_args = len(arg_types)

    @property
    def n_args(self) -> int:
        return self._n_args

    @classmethod
    def category_type(cls):
        return cls

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> torch.Tensor:
        # 训练时不需要真实计算，返回全零张量以通过流程
        device = data.data.device if hasattr(data, 'data') else torch.device('cpu')
        return torch.zeros((period.stop - period.start, data.n_stocks), device=device)

    def __str__(self):
        return self.name

# ====================================================
# 5. 实例化算子并精确计算 Action 空间偏移量
# ====================================================

# 实例化所有算子
OPERATORS = [CustomOperator(name, args, ret) for name, (args, ret) in OPERATOR_SIGNATURES.items()]

# 定义 Action 空间的物理结构
SIZE_NULL = 1                 # ID 0: 通常保留或作为空操作
SIZE_OP = len(OPERATORS)      # 算子数量
SIZE_FEATURE = len(FIELD_NAMES) # 特征数量 (1744)
SIZE_SEP = 1                  # 停止符数量

# 计算各个区间的起始偏移量
OFFSET_OP = SIZE_NULL              # 算子起始：1
OFFSET_FEATURE = OFFSET_OP + SIZE_OP  # 特征起始：1 + 算子数
OFFSET_SEP = OFFSET_FEATURE + SIZE_FEATURE # SEP起始：紧跟在最后一个特征后面

# Agent 最终看到的 Discrete 动作空间大小
SIZE_ALL = OFFSET_SEP + SIZE_SEP
SIZE_ACTION = SIZE_ALL - SIZE_NULL 

# ====================================================
# 6. 环境奖励相关
# ====================================================
REWARD_PER_STEP = 0.

# 打印核心配置摘要，方便启动时核对 ID 是否对齐
print(f"🚀 [Config] Logic Initialized:")
print(f"   - Operators : {SIZE_OP} (IDs: {OFFSET_OP} to {OFFSET_FEATURE-1})")
print(f"   - Features  : {SIZE_FEATURE} (IDs: {OFFSET_FEATURE} to {OFFSET_SEP-1})")
print(f"   - SEP ID    : {OFFSET_SEP-1} (Total Action Space: {SIZE_ACTION})")
print(f"   - Constants/DeltaTimes: Removed (Using Hardcoded Ops)")