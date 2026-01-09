from enum import IntEnum
from typing import Type, Union
from alphagen.data.expression import Operator

# 导入自定义字段
try:
    from adapters.字段字典_lol import result_dict as FIELD_DICT
    CUSTOM_FIELDS_AVAILABLE = True
    # 使用所有可用的字段（确保只使用真实存在的）
    FIELD_NAMES = list(FIELD_DICT.keys())
    FIELD_TEMPLATES = {}  # 模板字段 -> (基础类型, 填充选项)

    # 生成所有字段名，包括模板字段的展开
    FIELD_NAMES = []
    FIELD_TEMPLATES = {}  # 模板字段 -> (基础类型, 填充选项)

    # 解析字段和模板
    for field_name, (base_type, fill_options) in FIELD_DICT.items():
        if '_' in field_name and fill_options and isinstance(fill_options, list):
            # 是模板字段，记录模板信息
            FIELD_TEMPLATES[field_name] = (base_type, fill_options)

            # 计算需要填充的总位置数（下划线总数）
            total_positions = field_name.count('_')

            if total_positions != len(fill_options):
                print(f"Warning: Total underscore positions ({total_positions}) don't match fill options ({len(fill_options)}) for {field_name}")
                continue

            # 生成所有组合
            def generate_combinations(positions, current_parts=None):
                """递归生成所有填充组合"""
                if current_parts is None:
                    current_parts = []

                if len(current_parts) == len(positions):
                    # 所有位置都已填充，构建最终名称
                    result_name = field_name
                    for part in current_parts:
                        # 每次替换一个下划线
                        result_name = result_name.replace('_', part, 1)
                    FIELD_NAMES.append(result_name)
                    return

                # 填充下一个位置
                current_pos = len(current_parts)
                for option in positions[current_pos]:
                    generate_combinations(positions, current_parts + [option])

            generate_combinations(fill_options)
        else:
            # 普通字段，直接添加
            FIELD_NAMES.append(field_name)

    print(f"Loaded {len(FIELD_NAMES)} fields, {len(FIELD_TEMPLATES)} templates from 字段字典_lol.py")
except ImportError:
    FIELD_DICT = {}
    CUSTOM_FIELDS_AVAILABLE = False
    FIELD_NAMES = ['open', 'close', 'high', 'low', 'volume', 'vwap']  # 默认字段
    FIELD_TEMPLATES = {}

except ImportError:
    FIELD_DICT = {}
    CUSTOM_FIELDS_AVAILABLE = False
    FIELD_NAMES = ['open', 'close', 'high', 'low', 'volume', 'vwap']  # 默认字段
    FIELD_TEMPLATES = {}


class SequenceIndicatorType(IntEnum):
    BEG = 0
    SEP = 1


class Token:
    def __repr__(self):
        return str(self)


class ConstantToken(Token):
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def __str__(self): return str(self.constant)


class DeltaTimeToken(Token):
    def __init__(self, delta_time: int) -> None:
        self.delta_time = delta_time

    def __str__(self): return str(self.delta_time)


class FeatureToken(Token):
    def __init__(self, feature_name: str) -> None:
        self.feature_name = feature_name

    def __str__(self): return '@' + self.feature_name


class OperatorToken(Token):
    def __init__(self, operator: Union[Type[Operator], str]) -> None:
        self.operator = operator

    def __str__(self):
        if isinstance(self.operator, str):
            return self.operator
        # 对于Lorentz算子，使用name属性
        if hasattr(self.operator, 'name'):
            return self.operator.name
        return self.operator.__name__


class SequenceIndicatorToken(Token):
    def __init__(self, indicator: SequenceIndicatorType) -> None:
        self.indicator = indicator

    def __str__(self): return self.indicator.name


BEG_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.BEG)
SEP_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.SEP)
