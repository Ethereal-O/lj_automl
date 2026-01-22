#!/usr/bin/env python3
"""
Debug script to check SEP action availability logic
"""
import sys
import numpy as np

# Add current directory to path
sys.path.insert(0, '.')

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.models.alpha_pool import AlphaPool
from alphagen_qlib.calculator import ExternalCalculator

def debug_sep_logic():
    print("🔧 初始化环境...")

    # Create a dummy calculator
    def dummy_calculator(expr_str):
        return np.random.randn(10, 50), None, None

    calculator = ExternalCalculator(device='cpu', external_func=dummy_calculator)
    pool = AlphaPool(capacity=3, calculator=calculator, device='cpu')
    env = AlphaEnv(pool)

    print("📊 重置环境...")
    state, info = env.reset()

    print("🎮 开始调试SEP逻辑...\n")

    # Manually build a simple expression and check SEP availability
    from alphagen.data.tokens import FeatureToken

    # Add a single field
    print("步骤 1: 添加一个字段")
    field_action = None
    for i in range(len(env.action_masks())):
        if env.action(i).__class__.__name__ == 'FeatureToken':
            field_action = i
            break

    if field_action is not None:
        print(f"选择字段动作: {field_action}")
        next_state, reward, done, truncated, info = env.step(field_action)
        print(f"执行后 - 完成: {done}, 奖励: {reward:.4f}")

        # Check stack state
        stack = env.env._builder.stack
        print(f"栈大小: {len(stack)}")
        if stack:
            print(f"栈内容: {[str(item) for item in stack]}")
            print(f"栈元素类型: {[type(item).__name__ for item in stack]}")

        # Check SEP availability
        masks = env.action_masks()
        sep_available = masks[-1]  # SEP is last action
        print(f"SEP动作可用: {sep_available}")

        # Check _can_stop_by_validator result
        can_stop = env._can_stop_by_validator()
        print(f"_can_stop_by_validator() 返回: {can_stop}")

        if stack:
            single_part = stack[0]
            print(f"单元素是否featured: {getattr(single_part, 'is_featured', False)}")
            part_type = env._infer_expression_type(single_part)
            print(f"单元素类型: {part_type}")

if __name__ == "__main__":
    debug_sep_logic()
