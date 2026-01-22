#!/usr/bin/env python3
"""
测试严格的SEP停止逻辑
验证：只有当逆波兰式翻译后只剩一个元素且为数值类型时才允许停止
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphagen.rl.env.wrapper import AlphaEnvWrapper, AlphaEnvCore
from alphagen.models.alpha_pool import AlphaPool
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.calculator import AlphaCalculator
from alphagen.utils.random import reseed_everything
from adapters.field_config import field_config
from adapters.operator_library import OPERATOR_SIGNATURES
from alphagen.config import *
import numpy as np

def test_strict_sep_logic():
    """测试严格的SEP停止逻辑"""
    print("🧪 测试严格的SEP停止逻辑")
    print("=" * 60)
    
    # 1. 创建环境
    print("🔧 初始化环境...")
    
    # 创建一个简单的池子用于测试
    class MockCalculator(AlphaCalculator):
        def __init__(self):
            pass
        def calc_single_IC_ret(self, expr: Expression) -> float:
            return 0.0
        def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
            return 0.0
        def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
            return 0.0
        def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
            return 0.0
    
    pool = AlphaPool(capacity=10, calculator=MockCalculator())
    env_core = AlphaEnvCore(pool=pool, print_expr=True)
    env = AlphaEnvWrapper(env_core)
    
    # 2. 测试场景1：单个字段 - 应该允许停止
    print("\n📝 测试场景1：单个字段")
    print("-" * 40)
    obs, info = env.reset()
    
    # 添加一个字段
    field_token = FeatureToken(field_config.get_field_names()[0])  # 使用第一个字段
    # 直接使用动作索引，而不是从_tokens中查找
    action_index = OFFSET_FEATURE - 1 + 0  # 第一个字段的索引
    obs, reward, done, truncated, info = env.step(action_index)
    
    # 检查是否可以停止
    masks = env.action_masks()
    sep_allowed = masks[env_core.unwrapped.sep_action]
    print(f"字段: {field_token}")
    print(f"SEP允许: {sep_allowed}")
    print(f"栈状态: {len(env_core._builder.stack)} 个元素")
    if len(env_core._builder.stack) > 0:
        expr_type = env._get_expr_type(env_core._builder.stack[0])
        print(f"表达式类型: {expr_type}")
    
    # 3. 测试场景2：两个字段 - 不应该允许停止
    print("\n📝 测试场景2：两个字段")
    print("-" * 40)
    obs, info = env.reset()
    
    # 添加两个字段
    field_names = field_config.get_field_names()
    field1 = FeatureToken(field_names[0])
    field2 = FeatureToken(field_names[1])
    
    # 直接使用动作索引
    action_index1 = OFFSET_FEATURE - 1 + 0  # 第一个字段
    action_index2 = OFFSET_FEATURE - 1 + 1  # 第二个字段
    obs, reward, done, truncated, info = env.step(action_index1)
    obs, reward, done, truncated, info = env.step(action_index2)
    
    masks = env.action_masks()
    sep_allowed = masks[env_core.unwrapped.sep_action]
    print(f"字段1: {field1}")
    print(f"字段2: {field2}")
    print(f"SEP允许: {sep_allowed}")
    print(f"栈状态: {len(env_core._builder.stack)} 个元素")
    
    # 4. 测试场景3：简单算子表达式 - 应该允许停止
    print("\n📝 测试场景3：简单算子表达式")
    print("-" * 40)
    obs, info = env.reset()
    
    # 构建一个简单的表达式：TsMean5F(@field1)
    field_token = FeatureToken(field_config.get_field_names()[0])
    
    # 先添加字段
    action_index = OFFSET_FEATURE - 1 + 0  # 第一个字段
    obs, reward, done, truncated, info = env.step(action_index)
    
    # 找到TsMean5F算子
    ts_mean_op = None
    for i, op in enumerate(OPERATORS):
        if hasattr(op, 'name') and op.name == 'TsMean5F':
            ts_mean_op = op
            break
    
    if ts_mean_op:
        op_token = OperatorToken(ts_mean_op)
        action_index = OFFSET_OP - 1 + i  # 算子索引
        obs, reward, done, truncated, info = env.step(action_index)
        
        masks = env.action_masks()
        sep_allowed = masks[env_core.unwrapped.sep_action]
        print(f"表达式: {op_token}({field_token})")
        print(f"SEP允许: {sep_allowed}")
        print(f"栈状态: {len(env_core._builder.stack)} 个元素")
        if len(env_core._builder.stack) > 0:
            expr_type = env._get_expr_type(env_core._builder.stack[0])
            print(f"表达式类型: {expr_type}")
    else:
        print("❌ 未找到TsMean5F算子")
    
    # 5. 测试场景4：类型不匹配的算子 - 不应该允许停止
    print("\n📝 测试场景4：类型不匹配的算子")
    print("-" * 40)
    obs, info = env.reset()
    
    # 添加一个字段
    field_token = FeatureToken(field_config.get_field_names()[0])
    action_index = OFFSET_FEATURE - 1 + 0  # 第一个字段
    obs, reward, done, truncated, info = env.step(action_index)
    
    # 找一个需要vector输入的算子
    vector_op = None
    vector_op_index = None
    for i, op in enumerate(OPERATORS):
        op_name = getattr(op, 'name', op.__class__.__name__)
        if op_name in OPERATOR_SIGNATURES:
            arg_types, return_type = OPERATOR_SIGNATURES[op_name]
            if arg_types and arg_types[0] == 'vector':
                vector_op = op
                vector_op_index = i
                break
    
    if vector_op:
        op_token = OperatorToken(vector_op)
        try:
            action_index = OFFSET_OP - 1 + vector_op_index  # 算子索引
            obs, reward, done, truncated, info = env.step(action_index)
            
            masks = env.action_masks()
            sep_allowed = masks[env_core.unwrapped.sep_action]
            print(f"表达式: {op_token}({field_token})")
            print(f"SEP允许: {sep_allowed}")
            print(f"栈状态: {len(env_core._builder.stack)} 个元素")
            if len(env_core._builder.stack) > 0:
                expr_type = env._get_expr_type(env_core._builder.stack[0])
                print(f"表达式类型: {expr_type}")
        except Exception as e:
            print(f"❌ 算子应用失败: {e}")
    else:
        print("❌ 未找到需要vector输入的算子")
    
    # 6. 测试场景5：多个并列表达式 - 不应该允许停止
    print("\n📝 测试场景5：多个并列表达式")
    print("-" * 40)
    obs, info = env.reset()
    
    # 添加两个字段，形成并列表达式
    field_names = field_config.get_field_names()
    field1 = FeatureToken(field_names[0])
    field2 = FeatureToken(field_names[1])
    
    # 直接使用动作索引
    action_index1 = OFFSET_FEATURE - 1 + 0  # 第一个字段
    action_index2 = OFFSET_FEATURE - 1 + 1  # 第二个字段
    obs, reward, done, truncated, info = env.step(action_index1)
    obs, reward, done, truncated, info = env.step(action_index2)
    
    masks = env.action_masks()
    sep_allowed = masks[env_core.unwrapped.sep_action]
    print(f"并列表达式: {field1}, {field2}")
    print(f"SEP允许: {sep_allowed}")
    print(f"栈状态: {len(env_core._builder.stack)} 个元素")
    
    print("\n" + "=" * 60)
    print("✅ 严格SEP逻辑测试完成")
    print("🎯 验证要点：")
    print("   - 单个数值表达式：允许停止")
    print("   - 多个并列表达式：不允许停止")
    print("   - 类型不匹配：不允许停止")
    print("   - 栈中元素数量：必须为1")

if __name__ == "__main__":
    test_strict_sep_logic()
