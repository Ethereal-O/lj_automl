import os
import sys

# 添加项目根目录到路径，以便导入我们的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

# 导入我们的算子规则和字段字典
try:
    from adapters.算子规则 import OPERATOR_SIGNATURES, CONSTANT_RANGES
    from adapters.字段字典_lol import result_dict
    USE_CUSTOM_OPERATORS = True
    print("Using custom operators from 算子规则.py and 字段字典_lol.py")
except ImportError:
    print("Warning: Could not import custom operators, using default operators")
    USE_CUSTOM_OPERATORS = False

if USE_CUSTOM_OPERATORS:
    print(f"Total operators in 算子规则: {len(OPERATOR_SIGNATURES)}")

    # 创建自定义算子类，继承alphagen的算子基类
    from alphagen.data.expression import Expression, Operator, UnaryOperator, BinaryOperator, RollingOperator, PairRollingOperator, Constant
    import torch
    from alphagen_qlib.stock_data import StockData

    # 为不同参数数量创建专门的算子类
    class LorentzUnaryOperator(UnaryOperator):
        """Lorentz一元算子"""
        def __init__(self, name, arg_types, return_type):
            self.name = name
            self.arg_types = arg_types
            self.return_type = return_type
            # 初始化父类，但不传入operand（稍后在树构建时设置）
            super().__init__(0.0)  # 临时值，会被覆盖

        def _apply(self, operand: torch.Tensor) -> torch.Tensor:
            # 占位符实现 - 实际计算由lorentz程序执行
            return torch.zeros_like(operand)

        def __str__(self):
            return f"{self.name}"

    class LorentzBinaryOperator(BinaryOperator):
        """Lorentz二元算子"""
        def __init__(self, name, arg_types, return_type):
            self.name = name
            self.arg_types = arg_types
            self.return_type = return_type
            # 初始化父类，但不传入参数（稍后在树构建时设置）
            super().__init__(0.0, 0.0)  # 临时值，会被覆盖

        def _apply(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            # 占位符实现 - 实际计算由lorentz程序执行
            return torch.zeros_like(lhs)

        def __str__(self):
            return f"{self.name}"

    class LorentzRollingOperator(RollingOperator):
        """Lorentz滚动算子"""
        def __init__(self, name, arg_types, return_type):
            self.name = name
            self.arg_types = arg_types
            self.return_type = return_type
            # 初始化父类，但不传入参数（稍后在树构建时设置）
            super().__init__(0.0, 1)  # 临时值，会被覆盖

        def _apply(self, operand: torch.Tensor) -> torch.Tensor:
            # 占位符实现 - 实际计算由lorentz程序执行
            return torch.zeros((operand.shape[0], operand.shape[1]), dtype=operand.dtype, device=operand.device)

        def __str__(self):
            return f"{self.name}"

    class LorentzPairRollingOperator(PairRollingOperator):
        """Lorentz成对滚动算子"""
        def __init__(self, name, arg_types, return_type):
            self.name = name
            self.arg_types = arg_types
            self.return_type = return_type
            # 初始化父类，但不传入参数（稍后在树构建时设置）
            super().__init__(Constant(0.0), Constant(0.0), 1)  # 临时值，会被覆盖

        def _apply(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            # 占位符实现 - 实际计算由lorentz程序执行
            return torch.zeros((lhs.shape[0], lhs.shape[1]), dtype=lhs.dtype, device=lhs.device)

        def __str__(self):
            return f"{self.name}"

    class LorentzTernaryOperator(Operator):
        """Lorentz三元算子"""
        def __init__(self, name, arg_types, return_type):
            self.name = name
            self.arg_types = arg_types
            self.return_type = return_type
            self._n_args = len(arg_types)

        @classmethod
        def n_args(cls):
            return 3  # 默认3个参数

        @classmethod
        def category_type(cls):
            return PairRollingOperator  # 临时使用PairRollingOperator作为兼容性hack

        def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> torch.Tensor:
            # 占位符实现 - 实际计算由lorentz程序执行
            device = data.data.device if hasattr(data, 'data') else torch.device('cpu')
            dtype = torch.float32
            return torch.zeros((period.stop - period.start, data.n_stocks), dtype=dtype, device=device)

        def __str__(self):
            return f"{self.name}"

        @property
        def is_featured(self):
            return True

    # 创建所有算子的实例
    CUSTOM_OPERATORS = []
    for op_name, (arg_types, return_type) in OPERATOR_SIGNATURES.items():
        n_args = len(arg_types)

        if n_args == 1:
            op_class = LorentzUnaryOperator
        elif n_args == 2:
            # 简单检查是否是rolling算子（有时间相关参数）
            if any('const_int' in str(t) and ('time' in op_name.lower() or 'delta' in op_name.lower()) for t in arg_types):
                op_class = LorentzRollingOperator
            else:
                op_class = LorentzBinaryOperator
        elif n_args == 3:
            op_class = LorentzPairRollingOperator
        else:
            op_class = LorentzTernaryOperator  # 对于更多参数的算子

        op_instance = op_class(op_name, arg_types, return_type)
        CUSTOM_OPERATORS.append(op_instance)

    OPERATORS = CUSTOM_OPERATORS
    print(f"Using all {len(OPERATORS)} operators from 算子规则.py (lorentz implementation)")

    # 从取值范围中提取常量
    CUSTOM_CONSTANTS = []
    for range_name, range_values in CONSTANT_RANGES.items():
        if isinstance(range_values, (list, tuple)) and len(range_values) >= 2:
            # 从范围中生成一些常量值
            start, end = range_values[0], range_values[1]
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                # 生成一些中间值
                CUSTOM_CONSTANTS.extend([
                    float(start),
                    float((start + end) / 2),
                    float(end)
                ])

    # 去重并限制数量
    CUSTOM_CONSTANTS = list(set(CUSTOM_CONSTANTS))
    if CUSTOM_CONSTANTS:
        CONSTANTS = CUSTOM_CONSTANTS[:20]  # 限制常量数量
        print(f"Using {len(CONSTANTS)} constants from 算子规则 ranges")
    else:
        CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]
        print("Using default constants")

else:
    # 默认配置（保持兼容性）
    from alphagen.data.expression import *

    OPERATORS = [
        # Unary
        Abs, Log,
        # Binary
        Add, Sub, Mul, Div, Greater, Less,
        # Rolling
        Ref, Mean, Sum, Std, Var, Max, Min,
        Med, Mad, Delta, WMA, EMA,
        # Pair rolling
        Cov, Corr
    ]

    CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

DELTA_TIMES = [10, 20, 30, 40, 50]
REWARD_PER_STEP = 0.

# 为了向后兼容，提供这些常量（实际定义在alphagen.rl.env.wrapper中）
try:
    from alphagen.rl.env.wrapper import SIZE_OP, OFFSET_OP, OFFSET_FEATURE, OFFSET_SEP
except ImportError:
    SIZE_OP = len(OPERATORS) if 'OPERATORS' in globals() else 0
    OFFSET_OP = 1
    OFFSET_FEATURE = OFFSET_OP + SIZE_OP
    OFFSET_SEP = OFFSET_FEATURE + 291  # 假设291个字段
