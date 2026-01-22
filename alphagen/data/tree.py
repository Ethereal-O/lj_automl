from alphagen.data.expression import *
from alphagen.data.tokens import *


class ExpressionBuilder:
    stack: List[Expression]

    def __init__(self):
        self.stack = []

    def get_tree(self) -> Expression:
        if len(self.stack) == 1:
            return self.stack[0]
        else:
            raise InvalidExpressionException(f"Expected only one tree, got {len(self.stack)}")

    def add_token(self, token: Token):
        # 检查是否是语法学习阶段
        import os
        is_syntax_learning = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'

        if not self.validate(token):
            # 无论语法学习还是IC学习，都抛出异常让validator处理
            # validator会检测irrecoverable状态并立即结束episode
            raise InvalidExpressionException(f"Token {token} not allowed here, stack: {self.stack}.")
        if isinstance(token, OperatorToken):
            n_args: int = token.operator.n_args()
            children = []
            for _ in range(n_args):
                children.append(self.stack.pop())

            # 对于Lorentz算子，需要特殊处理实例化
            if hasattr(token.operator, 'name'):  # Lorentz算子
                # 创建新的算子实例而不是直接调用构造函数
                op_class = token.operator.__class__
                op_instance = op_class(token.operator.name, token.operator.arg_types, token.operator.return_type)
                # 设置参数（逆波兰式从右到左）
                if n_args == 1:
                    op_instance._operand = children[0]
                elif n_args == 2:
                    op_instance._lhs = children[1]
                    op_instance._rhs = children[0]
                elif n_args == 3:
                    op_instance._lhs = children[2]
                    op_instance._rhs = children[1]
                    op_instance._delta_time = children[0]._delta_time if hasattr(children[0], '_delta_time') else children[0]

                self.stack.append(op_instance)
            else:
                # 标准alphagen算子
                self.stack.append(token.operator(*reversed(children)))  # type: ignore

        elif isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
        elif isinstance(token, DeltaTimeToken):
            self.stack.append(DeltaTime(token.delta_time))
        elif isinstance(token, FeatureToken):
            # 处理字符串字段名
            if hasattr(token, 'feature_name'):
                # 创建一个简单的特征节点（实际计算由lorentz处理）
                class LorentzFeature:
                    def __init__(self, name):
                        self.name = name
                    def __str__(self):
                        return f"@{self.name}"
                    @property
                    def is_featured(self):
                        return True
                    def evaluate(self, data, period=None):
                        # 对于Lorentz特征，使用外部计算
                        import subprocess
                        from io import StringIO
                        import pandas as pd
                        import numpy as np
                        import torch

                        expr_str = f"@{self.name}"

                        try:
                            # 调用external_compute.py
                            result = subprocess.run(['python3', 'external_compute.py', expr_str],
                                                  capture_output=True, text=True, cwd='.')
                            if result.returncode == 0:
                                # 解析输出
                                df = pd.read_csv(StringIO(result.stdout))
                                if not df.empty and 'factor_value' in df.columns:
                                    # 转换为张量格式
                                    device = data.data.device if hasattr(data, 'data') else torch.device('cpu')
                                    # 这里需要根据实际数据格式调整
                                    values = torch.tensor(df['factor_value'].values, dtype=torch.float32, device=device)
                                    # 重新形状为 (n_days, n_stocks)
                                    if len(values.shape) == 1:
                                        values = values.unsqueeze(-1)  # 添加股票维度
                                    return values
                        except Exception as e:
                            print(f"External computation failed for {expr_str}: {e}")

                        # fallback: 返回零张量
                        device = data.data.device if hasattr(data, 'data') else torch.device('cpu')
                        return torch.zeros((period.stop - period.start, data.n_stocks), dtype=torch.float32, device=device)

                self.stack.append(LorentzFeature(token.feature_name))
            else:
                # 兼容旧的FeatureType
                self.stack.append(Feature(token.feature))

        else:
            assert False

    def is_valid(self) -> bool:
        """检查是否是完整有效的表达式（向后兼容）"""
        return len(self.stack) == 1 and self.stack[0].is_featured

    def is_complete_expression(self) -> bool:
        """
        检查当前状态是否是完整的可计算表达式
        即：可以直接用于因子值计算的状态
        """
        return self.is_valid()

    def is_intermediate_valid(self) -> bool:
        """
        检查当前状态是否是合理的中间表达式
        即：栈中所有元素都是featured的完整部分，可以继续添加算子或字段
        """
        if len(self.stack) == 0:
            return False  # 空栈不是有效的中间状态

        # 检查栈中所有元素是否都是featured（完整部分）
        for expr in self.stack:
            if not expr.is_featured:
                return False

        # 简化的类型检查，避免使用废弃文件
        try:
            for expr in self.stack:
                # 使用简化的类型推断
                expr_type = self._simple_infer_type(expr)

                # 检查是否是数值类型（float及其子类）
                numeric_types = ['float', 'int']
                if expr_type not in numeric_types:
                    return False

            return True

        except Exception:
            return False

    def is_irrecoverably_invalid(self) -> bool:
        """
        检查当前状态是否真正不可恢复的无效
        只检测无法通过任何后续操作修复的错误状态
        """
        # 检查结构限制
        if len(self.stack) > 20:  # 太深的嵌套
            return True

        # 检查是否有连续的DeltaTime（通常是错误的）
        consecutive_dt = 0
        for item in self.stack:
            if isinstance(item, DeltaTime):
                consecutive_dt += 1
                if consecutive_dt >= 2:
                    return True
            else:
                consecutive_dt = 0

        # 检查是否有未匹配的操作符参数（真正的结构错误）
        try:
            # 更精确的检查：如果有操作符但参数明显不足
            # 注意：这里不应该把正常的"多个并列完整部分"当作错误
            # 因为它们可以通过二元算子连接

            # 只检查极端情况：如只有一个操作符但没有操作数
            if len(self.stack) == 1 and isinstance(self.stack[0], Operator):
                return True  # 单个操作符无法恢复

            # 检查是否有操作符的参数明显不足的情况
            # 但不把正常的并列表达式当作错误
            operator_count = sum(1 for item in self.stack if isinstance(item, Operator))
            if operator_count > 0:
                # 计算理论上需要的参数总数
                total_args_needed = 0
                for item in self.stack:
                    if isinstance(item, Operator):
                        # 简化估算：假设每个操作符需要2个参数（实际可能更多）
                        total_args_needed += 2

                # 如果操作数明显不足，才认为是不可恢复的
                operands_count = len(self.stack) - operator_count
                if operands_count < operator_count:  # 参数严重不足
                    return True

        except:
            pass

        # 不检查类型兼容性作为IRRECOVERABLE的条件
        # 因为类型问题可以通过正确的算子选择解决
        # 让RL通过学习发现正确的组合

        return False

    def get_expression_state(self) -> str:
        """
        获取当前表达式的精细状态
        返回: 'complete', 'intermediate_valid', 'intermediate_invalid', 'irrecoverable'
        """
        if self.is_complete_expression():
            return 'complete'
        elif self.is_irrecoverably_invalid():
            return 'irrecoverable'
        elif self.is_intermediate_valid():
            return 'intermediate_valid'
        else:
            return 'intermediate_invalid'

    def validate(self, token: Token) -> bool:
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator)
        elif isinstance(token, DeltaTimeToken):
            return self.validate_dt()
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, FeatureToken):
            return self.validate_feature()
        else:
            assert False

    def validate_op(self, op) -> bool:
        # 检查是否是语法学习阶段
        import os
        is_syntax_learning = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'

        # 处理Lorentz算子实例
        if hasattr(op, 'n_args'):
            n_args = op.n_args()
        else:
            # 标准alphagen算子类
            n_args = op.n_args()

        if len(self.stack) < n_args:
            print(f"DEBUG validate_op: {op.name if hasattr(op, 'name') else op} rejected - insufficient stack depth {len(self.stack)} < {n_args}")
            return False

        # 对于Lorentz算子，进行类型检查
        if hasattr(op, 'name'):  # Lorentz算子
            # 检查栈顶是否有足够的featured元素
            for i in range(n_args):
                expr = self.stack[-(i+1)] if len(self.stack) > i else None
                if expr is None or not expr.is_featured:
                    print(f"DEBUG validate_op: {op.name} rejected - param {i} not featured: {expr}")
                    return False

            # 对于 Lorentz 算子，检查参数类型是否匹配
            if hasattr(op, 'arg_types') and op.arg_types:
                expected_types = op.arg_types

                for i in range(n_args):
                    stack_expr = self.stack[-(i+1)]
                    expected_type = expected_types[i] if i < len(expected_types) else expected_types[-1]

                    # 简化的类型推断（与 action_masks 保持一致）
                    actual_type = self._simple_infer_type(stack_expr)

                    # 简化的类型兼容性检查
                    if not self._simple_type_compatible(actual_type, expected_type):
                        print(f"DEBUG validate_op: {op.name} rejected - type mismatch param {i}: {actual_type} vs {expected_type}")
                        return False

            print(f"DEBUG validate_op: {op.name} accepted")
            return True

        # 标准alphagen算子类的验证逻辑
        if issubclass(op, UnaryOperator):
            if not self.stack[-1].is_featured:
                return False
        elif issubclass(op, BinaryOperator):
            if not self.stack[-1].is_featured and not self.stack[-2].is_featured:
                return False
            if (isinstance(self.stack[-1], DeltaTime) or
                    isinstance(self.stack[-2], DeltaTime)):
                return False
        elif issubclass(op, RollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured:
                return False
        elif issubclass(op, PairRollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured or not self.stack[-3].is_featured:
                return False
        else:
            # 对于其他类型的算子，允许通过（Lorentz算子的情况）
            pass
        return True

    def _simple_infer_type(self, expr) -> str:
        """
        简化的类型推断，与 action_masks 中的 _infer_type 保持一致
        """
        # 1. 如果是 LorentzFeature (特殊处理)
        if hasattr(expr, 'name') and str(expr).startswith('@'):
            # LorentzFeature 的 name 就是特征名，如 "Preload.SW3Size"
            feat_name = expr.name
            try:
                from adapters.dic_lol import result_dict
                if feat_name in result_dict:
                    return result_dict[feat_name][0]  # (type, choices)
            except:
                pass
            return "float"  # 默认

        # 2. 如果是 Feature (叶子节点)
        if hasattr(expr, 'feature'):
            # 处理 FeatureToken 或 FeatureExpression
            feat_name = str(expr.feature).replace("Feature.", "").replace("@", "").strip("'\"")
            # 使用 result_dict 进行类型映射
            try:
                from adapters.dic_lol import result_dict
                if feat_name in result_dict:
                    return result_dict[feat_name][0]  # (type, choices)
            except:
                pass
            return "float"  # 默认

        # 3. 如果是 Constant (常量)
        if hasattr(expr, '_value'):
            if isinstance(expr._value, int):
                return "int"
            return "float"

        # 4. 如果是 Operator (算子节点)
        op_name = getattr(expr, 'name', expr.__class__.__name__)
        try:
            from adapters.operator_library import OPERATOR_SIGNATURES
            if op_name in OPERATOR_SIGNATURES:
                _, return_type = OPERATOR_SIGNATURES[op_name]
                return return_type
        except:
            pass

        # 5. 未知类型，保守返回float
        return 'float'

    def _simple_type_compatible(self, actual_type: str, expected_type: str) -> bool:
        """
        简化的类型兼容性检查，与 action_masks 中的逻辑保持一致
        """
        if expected_type in ['any', 'expr']:
            return True
        if actual_type == expected_type:
            return True
        if expected_type == "vector" and "vector" in actual_type:
            return True
        if expected_type == "float" and actual_type == "int":
            return True
        return False

    def validate_dt(self) -> bool:
        return len(self.stack) > 0 and self.stack[-1].is_featured

    def validate_const(self) -> bool:
        return len(self.stack) == 0 or self.stack[-1].is_featured

    def validate_feature(self) -> bool:
        return not (len(self.stack) >= 1 and isinstance(self.stack[-1], DeltaTime))


class InvalidExpressionException(ValueError):
    pass


if __name__ == '__main__':
    tokens = [
        FeatureToken(FeatureType.LOW),
        OperatorToken(Abs),
        DeltaTimeToken(-10),
        OperatorToken(Ref),
        FeatureToken(FeatureType.HIGH),
        FeatureToken(FeatureType.CLOSE),
        OperatorToken(Div),
        OperatorToken(Add),
    ]

    builder = ExpressionBuilder()
    for token in tokens:
        builder.add_token(token)

    print(f'res: {str(builder.get_tree())}')
    print(f'ref: Add(Ref(Abs($low),-10),Div($high,$close))')
