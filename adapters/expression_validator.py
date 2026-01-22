

"""
增强版表达式验证器
支持复杂的字段字典、多层嵌套选择、类型继承和兼容性评分
"""

from typing import Any, Dict, Tuple, List, Optional, Union
import re
import math

# 导入配置
try:
    from adapters import dic_lol as field_dict_module
    from adapters import rule as operator_rules_module

    FIELD_DICT = field_dict_module.result_dict
    OPERATOR_SIGNATURES = operator_rules_module.OPERATOR_SIGNATURES
    TYPE_HIERARCHY = operator_rules_module.TYPE_HIERARCHY
    get_type_compatibility = operator_rules_module.get_type_compatibility
    RANGE_HINTS = operator_rules_module.RANGE_HINTS
    RANGE_COMPATIBILITY = operator_rules_module.RANGE_COMPATIBILITY
    CONSTANT_RANGES = operator_rules_module.CONSTANT_RANGES
    # VALUE_RANGE_GROUPS在rule中不存在，暂时设为空
    VALUE_RANGE_GROUPS = getattr(operator_rules_module, 'VALUE_RANGE_GROUPS', {})

except ImportError as e:
    print(f"导入配置失败: {e}")
    # 如果文件不存在，使用简化版本
    FIELD_DICT = {}
    OPERATOR_SIGNATURES = {}
    TYPE_HIERARCHY = {}
    RANGE_HINTS = {}
    RANGE_COMPATIBILITY = {}
    CONSTANT_RANGES = {}
    VALUE_RANGE_GROUPS = {}

    def get_type_compatibility(expected, actual):
        return 1.0 if expected == actual else 0.0


def expand_field_patterns(field_dict: Dict[str, Tuple[str, List[List[str]]]]) -> Dict[str, str]:
    """
    将字段字典中的模式展开为具体的字段名

    Args:
        field_dict: 字段字典，包含模式定义

    Returns:
        展开后的字段到类型的映射
    """
    expanded = {}

    for pattern, (base_type, choices_list) in field_dict.items():
        if not choices_list:  # 简单字段
            expanded[pattern] = base_type
        else:  # 复杂模式，需要展开
            # 生成所有可能的组合
            combinations = _generate_combinations(choices_list)
            for combo in combinations:
                field_name = pattern
                for choice in combo:
                    if choice:  # 非空选择
                        field_name = field_name.replace('_', choice, 1)
                    else:  # 空选择，移除对应的_
                        field_name = re.sub(r'_([^_]*)', r'\1', field_name, count=1)

                # 处理剩余的_（如果有的话）
                field_name = field_name.replace('_', '')

                expanded[field_name] = base_type

    return expanded


def _generate_combinations(choices_list: List[List[str]]) -> List[Tuple[str, ...]]:
    """生成所有可能的组合"""
    if not choices_list:
        return [()]

    def _combine(current: List[List[str]], index: int) -> List[Tuple[str, ...]]:
        if index == len(current):
            return [()]

        result = []
        for choice in current[index]:
            for rest in _combine(current, index + 1):
                result.append((choice,) + rest)
        return result

    return _combine(choices_list, 0)


# 展开字段字典
EXPANDED_FIELD_DICT = expand_field_patterns(FIELD_DICT)


def parse_expression(expr_str: str) -> Any:
    """解析表达式，支持多种格式：函数调用、RPN中间状态、树字符串"""
    expr_str = expr_str.strip()

    # 处理函数调用格式：FuncName(arg1,arg2,...)
    func_pattern = r'(\w+)\(([^)]*)\)'
    match = re.match(func_pattern, expr_str)
    if match:
        func_name = match.group(1)
        args_str = match.group(2)

        # 解析参数
        if args_str:
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

            # 递归解析每个参数
            parsed_args = []
            for arg in args:
                parsed_args.append(parse_expression(arg))

            return {'op': func_name, 'args': parsed_args}
        else:
            return {'op': func_name, 'args': []}

    # 处理RPN中间状态：[token1, token2, ...]
    elif expr_str.startswith('[') and expr_str.endswith(']'):
        # 解析RPN token列表
        content = expr_str[1:-1].strip()
        if not content:
            return []  # 空列表

        # 分割tokens（简单按逗号分割）
        tokens = [t.strip().strip('\'"') for t in content.split(',') if t.strip()]
        return tokens

    # 处理ExpressionBuilder的树字符串格式
    elif expr_str.startswith('$') or 'Constant(' in expr_str:
        # 叶子节点：字段引用或常量
        return expr_str

    elif any(op in expr_str for op in OPERATOR_SIGNATURES.keys()):
        # 包含算子的复杂表达式，尝试解析
        # 这里可以扩展更复杂的解析逻辑
        return expr_str  # 暂时返回原字符串

    else:
        # 其他情况，当作叶子节点
        return expr_str


def parse_tokens(tokens: List[str]) -> Any:
    """递归解析tokens"""
    if not tokens:
        return None

    token = tokens.pop(0)
    if token == '(':
        op = tokens.pop(0)
        args = []
        while tokens and tokens[0] != ')':
            if tokens[0] == ',':
                tokens.pop(0)
                continue
            args.append(parse_tokens(tokens))
        if tokens and tokens[0] == ')':
            tokens.pop(0)
        return {'op': op, 'args': args}
    else:
        return token


def get_leaf_type(token: str) -> str:
    """获取叶子节点的类型"""
    # 检查是否是字段引用（支持$和@两种格式）
    if token.startswith('$') or token.startswith('@'):
        field_name = token[1:]
        if field_name in EXPANDED_FIELD_DICT:
            return EXPANDED_FIELD_DICT[field_name]
        else:
            return 'unknown'

    # 检查是否是常量
    elif token.startswith('Constant(') and token.endswith(')'):
        value = token[9:-1]
        try:
            float(value)
            return 'const_float'
        except:
            if value in ['True', 'False']:
                return 'const_bool'
            return 'unknown'

    # 检查是否是数字
    else:
        try:
            int(token)
            return 'const_int'
        except:
            try:
                float(token)
                return 'const_float'
            except:
                return 'unknown'


def validate_expression_types(tree: Any) -> Tuple[bool, str, float]:
    """
    递归验证类型并计算兼容性分数
    支持完整表达式树和RPN中间状态

    Returns:
        (is_valid, error_msg, compatibility_score)
    """
    # 处理RPN token列表（中间状态）
    if isinstance(tree, list):
        return validate_rpn_tokens(tree)

    # 处理完整表达式树
    if isinstance(tree, str):
        typ = get_leaf_type(tree)
        if typ == 'unknown':
            return False, f"Unknown token: {tree}", 0.0
        return True, typ, 1.0

    if isinstance(tree, dict):
        op = tree['op']
        args = tree['args']

        if op not in OPERATOR_SIGNATURES:
            return False, f"Unknown operator: {op}", 0.0

        arg_specs, return_type_spec = OPERATOR_SIGNATURES[op]

        # 检查参数数量
        if len(args) != len(arg_specs):
            return False, f"Operator {op} expects {len(arg_specs)} args, got {len(args)}", 0.0

        # 验证每个参数
        total_score = 0.0
        arg_types = []

        for i, arg in enumerate(args):
            valid, arg_type, arg_score = validate_expression_types(arg)
            if not valid:
                return False, f"Arg {i} invalid: {arg_type}", 0.0

            expected = arg_specs[i]
            compatibility = get_type_compatibility(expected, arg_type)
            total_score += compatibility
            arg_types.append(arg_type)

        # 计算平均兼容性分数
        avg_score = total_score / len(args) if args else 1.0

        # 确定返回类型
        if return_type_spec == "same_as_input":
            # 使用第一个参数的类型
            return_type = arg_types[0] if arg_types else "unknown"
        elif return_type_spec == "expr_return_type":
            # 根据上下文推断
            return_type = "expr_return_type"  # 简化处理
        else:
            return_type = return_type_spec

        return True, return_type, avg_score

    return False, "Invalid tree", 0.0


def validate_rpn_tokens(tokens: List[str]) -> Tuple[bool, str, float]:
    """
    验证RPN token序列是否是合理的中间状态

    Args:
        tokens: RPN token列表

    Returns:
        (is_valid, error_msg, compatibility_score)
    """
    if not tokens:
        return True, "Empty RPN sequence", 0.0

    stack = []
    total_score = 0.0
    score_count = 0

    for token in tokens:
        if token in OPERATOR_SIGNATURES:
            # 操作符：从栈中弹出参数，验证类型兼容性
            op_info = OPERATOR_SIGNATURES[token]
            arg_specs, return_type = op_info

            if len(stack) < len(arg_specs):
                return False, f"Operator {token} needs {len(arg_specs)} args, but stack has {len(stack)}", 0.0

            # 验证参数类型兼容性
            arg_types = []
            for i in range(len(arg_specs)):
                arg_type = stack.pop()
                expected = arg_specs[-(i+1)]  # 逆序检查
                compatibility = get_type_compatibility(expected, arg_type)
                total_score += compatibility
                score_count += 1
                arg_types.append(arg_type)

            # 确定返回类型并压栈
            if return_type == "same_as_input":
                result_type = arg_types[0] if arg_types else "unknown"
            elif return_type == "expr_return_type":
                result_type = "expr_return_type"
            else:
                result_type = return_type

            stack.append(result_type)

        elif token.startswith('$'):
            # 字段引用
            field_name = token[1:]
            if field_name in EXPANDED_FIELD_DICT:
                field_type = EXPANDED_FIELD_DICT[field_name]
                stack.append(field_type)
            else:
                return False, f"Unknown field: {field_name}", 0.0

        elif token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
            # 整数常量
            stack.append("const_int")

        elif _is_float(token):
            # 浮点常量
            stack.append("const_float")

        else:
            # 未知token
            return False, f"Unknown token: {token}", 0.0

    # 计算平均兼容性分数
    avg_score = total_score / score_count if score_count > 0 else 1.0

    # 检查栈状态
    if len(stack) == 1:
        # 只有一个元素，可能是完整表达式
        top_type = stack[0]
        if top_type in ['float', 'int', 'const_float', 'const_int']:
            return True, f"Complete expression (type: {top_type})", avg_score
        else:
            return False, f"Incomplete expression (top type: {top_type})", avg_score
    elif len(stack) > 1:
        # 多个元素，可能是中间状态
        return True, f"Intermediate state ({len(stack)} items on stack)", avg_score
    else:
        # 栈为空，可能是无效序列
        return False, "Invalid RPN sequence (empty stack)", 0.0


def _is_float(s: str) -> bool:
    """检查字符串是否是浮点数"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def calculate_matching_degree(expr_str: str) -> float:
    """
    计算表达式的匹配度（类型兼容性分数）
    用于RL奖励中的匹配度惩罚
    """
    try:
        tree = parse_expression(expr_str)
        if tree is None:
            return 0.0

        valid, output_type, score = validate_expression_types(tree)
        if not valid:
            return 0.0

        # 检查输出类型是否合适（应该产生数值类型）
        if output_type not in ['float', 'int', 'const_float', 'const_int']:
            # 如果输出不是数值类型，降低分数
            score *= 0.5

        return score

    except Exception as e:
        return 0.0


def validate_expression(expr_str: str) -> Tuple[bool, str]:
    """
    验证表达式是否有效
    对RPN中间状态和完整表达式采用不同的验证标准

    Returns:
        (is_valid, message)
    """
    try:
        tree = parse_expression(expr_str)
        if tree is None:
            return False, "Empty expression"

        valid, output_type, score = validate_expression_types(tree)
        if not valid:
            return False, output_type

        # 对不同类型的输入采用不同的验证标准
        if isinstance(tree, list):
            # RPN中间状态：只要结构合理即可，不要求最终输出数值
            return True, f"RPN intermediate: {output_type} (compatibility: {score:.2f})"
        else:
            # 完整表达式：要求输出数值类型
            if output_type in ['float', 'int', 'const_float', 'const_int']:
                return True, f"Complete expression: {output_type} (compatibility: {score:.2f})"
            else:
                return False, f"Output type {output_type} not suitable for factor"

    except Exception as e:
        return False, str(e)


def get_expression_complexity(expr_str: str) -> int:
    """
    计算表达式的复杂度（用于奖励设计）
    """
    try:
        tree = parse_expression(expr_str)
        return _calculate_tree_complexity(tree)
    except:
        return 0


def _calculate_tree_complexity(tree: Any) -> int:
    """递归计算树复杂度"""
    if isinstance(tree, str):
        return 1
    elif isinstance(tree, dict):
        complexity = 1  # 操作符本身
        for arg in tree['args']:
            complexity += _calculate_tree_complexity(arg)
        return complexity
    else:
        return 0


# 初始化时验证配置
if EXPANDED_FIELD_DICT:
    print(f"Loaded {len(EXPANDED_FIELD_DICT)} field definitions")
if OPERATOR_SIGNATURES:
    print(f"Loaded {len(OPERATOR_SIGNATURES)} operator signatures")
