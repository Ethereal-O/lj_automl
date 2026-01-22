#!/usr/bin/env python3
"""
Test script to verify ExpressionBuilder logic
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

from alphagen.data.tree import ExpressionBuilder
from alphagen.data.tokens import *
from alphagen.config import OPERATORS

def test_expression_builder():
    """
    Test ExpressionBuilder with various token sequences
    """
    print("🔧 测试ExpressionBuilder逻辑\n")

    # Test case 1: Single field
    print("=== 测试1: 单个字段 ===")
    builder = ExpressionBuilder()
    field_token = FeatureToken("@Slice.Close")
    builder.add_token(field_token)
    print(f"添加 {field_token} 后，stack大小: {len(builder.stack)}")
    print(f"stack内容: {[str(x) for x in builder.stack]}")
    print(f"is_complete_expression(): {builder.is_complete_expression()}")
    try:
        tree = builder.get_tree()
        print(f"get_tree()成功: {tree}")
    except Exception as e:
        print(f"get_tree()失败: {e}")
    print()

    # Test case 2: Field + Operator
    print("=== 测试2: 字段 + 单参数算子 ===")
    builder = ExpressionBuilder()
    builder.add_token(FeatureToken("@Slice.Close"))
    print(f"添加字段后，stack大小: {len(builder.stack)}")

    # Test with a simple unary operator from OPERATORS
    unary_op = None
    for op in OPERATORS:
        if hasattr(op, 'n_args') and op.n_args() == 1:
            unary_op = op
            break

    if unary_op:
        op_name = getattr(unary_op, 'name', str(unary_op))
        print(f"找到单参数算子: {op_name}")
        op_token = OperatorToken(unary_op)
        try:
            builder.add_token(op_token)
            print(f"添加 {op_name} 后，stack大小: {len(builder.stack)}")
            print(f"stack内容: {[str(x) for x in builder.stack]}")
            print(f"is_complete_expression(): {builder.is_complete_expression()}")
            try:
                tree = builder.get_tree()
                print(f"get_tree()成功: {tree}")
            except Exception as e:
                print(f"get_tree()失败: {e}")
        except Exception as e:
            print(f"添加算子失败: {e}")
    else:
        print("未找到合适的单参数算子")
    print()

    # Test case 3: Multiple parallel parts
    print("=== 测试3: 多个并列部分 ===")
    builder = ExpressionBuilder()
    builder.add_token(FeatureToken("@Slice.Close"))
    builder.add_token(FeatureToken("@Slice.Volume"))
    print(f"添加两个字段后，stack大小: {len(builder.stack)}")
    print(f"stack内容: {[str(x) for x in builder.stack]}")
    print(f"is_complete_expression(): {builder.is_complete_expression()}")
    try:
        tree = builder.get_tree()
        print(f"get_tree()成功: {tree}")
    except Exception as e:
        print(f"get_tree()失败: {e}")
    print()

    # Test case 4: Parallel parts with operators
    print("=== 测试4: 并列部分 + 算子 ===")
    builder = ExpressionBuilder()
    builder.add_token(FeatureToken("@Slice.Close"))
    builder.add_token(FeatureToken("@Slice.Volume"))
    if unary_op:
        op_name = getattr(unary_op, 'name', str(unary_op))
        op_token = OperatorToken(unary_op)
        try:
            builder.add_token(op_token)
            print(f"添加两个字段 + {op_name} 后，stack大小: {len(builder.stack)}")
            print(f"stack内容: {[str(x) for x in builder.stack]}")
            print(f"is_complete_expression(): {builder.is_complete_expression()}")
            try:
                tree = builder.get_tree()
                print(f"get_tree()成功: {tree}")
            except Exception as e:
                print(f"get_tree()失败: {e}")
        except Exception as e:
            print(f"添加算子失败: {e}")
    print()

if __name__ == "__main__":
    test_expression_builder()
