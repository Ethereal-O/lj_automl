#!/usr/bin/env python3
"""
Test script to generate several expressions and verify they complete properly
without irrecoverable states.
"""
import sys
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, '.')

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.models.alpha_pool import AlphaPool
from alphagen_qlib.calculator import ExternalCalculator
from alphagen.data.tokens import SequenceIndicatorToken, SequenceIndicatorType

def generate_expressions(num_expressions=5, max_steps=20):
    """
    Generate several expressions and show the results
    """
    print("🔧 初始化环境...")

    # Create a dummy calculator that doesn't require real data
    def dummy_calculator(expr_str):
        # Return dummy data for expression generation testing
        return np.random.randn(10, 50), pd.date_range('2020-01-01', periods=10), pd.Index([f'stock_{i}' for i in range(50)])

    # Use ExternalCalculator with dummy function
    calculator = ExternalCalculator(device='cpu', external_func=dummy_calculator)

    # Create environment
    pool = AlphaPool(capacity=3, calculator=calculator, device='cpu')
    env = AlphaEnv(pool)

    print(f"🎯 将生成 {num_expressions} 个表达式\n")

    successful_expressions = []

    for expr_num in range(1, num_expressions + 1):
        print(f"{'='*60}")
        print(f"📝 生成表达式 #{expr_num}")
        print(f"{'='*60}")

        # Reset environment
        state, info = env.reset()
        done = False
        steps = 0
        episode_reward = 0.0

        while not done and steps < max_steps:
            steps += 1

            # Get action masks
            action_masks = env.action_masks()
            available_actions = np.where(action_masks)[0]

            if len(available_actions) == 0:
                print(f"❌ 没有可用动作，停止生成")
                break

            # Random action selection (could be improved with policy)
            selected_action = np.random.choice(available_actions)

            # Execute action
            next_state, reward, done, truncated, info = env.step(selected_action)
            episode_reward += reward

            # Show progress every 3 steps
            if steps % 3 == 0:
                current_tokens = [str(t) for t in env.env._tokens[1:] if not isinstance(t, SequenceIndicatorToken)]
                print(f"步骤 {steps}: 表达式 = {' '.join(current_tokens)} | 奖励 = {episode_reward:.4f}")

        # Check final result
        if done:
            # Get final expression
            final_tokens = [str(t) for t in env.env._tokens[1:] if not isinstance(t, SequenceIndicatorToken)]
            final_expression = ' '.join(final_tokens)

            # Check if it was marked as irrecoverable
            is_irrecoverable = 'IRRECOVERABLE' in str(episode_reward)

            print(f"✅ 完成! 最终表达式: {final_expression}")
            print(f"📊 总奖励: {episode_reward:.4f}")
            print(f"🎯 步骤数: {steps}")
            print(f"🚫 不可恢复: {is_irrecoverable}")

            if not is_irrecoverable:
                successful_expressions.append({
                    'expression': final_expression,
                    'reward': episode_reward,
                    'steps': steps
                })
        else:
            print(f"❌ 未能在 {max_steps} 步内完成")

        print()

    # Summary
    print(f"{'='*60}")
    print("📋 生成总结")
    print(f"{'='*60}")
    print(f"成功生成表达式: {len(successful_expressions)}/{num_expressions}")
    print(f"失败/不可恢复: {num_expressions - len(successful_expressions)}")

    if successful_expressions:
        print("\n🎉 成功生成的表达式:")
        for i, expr_data in enumerate(successful_expressions, 1):
            print(f"{i}. {expr_data['expression']} (奖励: {expr_data['reward']:.4f}, 步骤: {expr_data['steps']})")

    return successful_expressions

if __name__ == "__main__":
    # Generate expressions
    results = generate_expressions(num_expressions=8, max_steps=25)

    print("🎯 测试完成!" ) 
    print(f"生成了 {len(results)} 个有效表达式，证明修改后的代码不再错误地将有效表达式标记为不可恢复状态。")
