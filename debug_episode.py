#!/usr/bin/env python3
"""
Debug script to run a single episode and show detailed step-by-step information
"""
import sys
import os
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, '.')

# 不设置ALPHAQCM_SYNTAX_LEARNING环境变量，使用真实计算

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.data.tokens import SequenceIndicatorToken, SequenceIndicatorType
from alphagen_qlib.calculator import TestStockDataCalculator
from alphagen_qlib.stock_data import StockData

def debug_single_episode():
    print("🔧 初始化环境...")

    # Create a dummy calculator for syntax learning (no real calculation needed)
    def dummy_calculator(expr_str):
        return 0.0  # Always return 0 for syntax learning

    from alphagen_qlib.calculator import ExternalCalculator
    calculator = ExternalCalculator(device='cpu', external_func=dummy_calculator)

    # Create environment
    pool = AlphaPool(capacity=3, calculator=calculator, device='cpu')
    env = AlphaEnv(pool)

    print("📊 重置环境...")
    state, info = env.reset()
    print(f"初始状态: {state}")

    print("\n🎮 开始单个episode...\n")

    episode_reward = 0.0
    step = 0
    done = False

    while not done and step < 10:  # Limit steps for debugging
        step += 1
        print(f"\n{'='*50}")
        print(f"📍 步骤 {step}")
        print(f"{'='*50}")

        # Get current expression state
        try:
            current_state = env.env._builder.get_expression_state()
            stack_size = len(env.env._builder.stack)
            print(f"当前表达式状态: {current_state}")
            print(f"栈大小: {stack_size}")
            if env.env._builder.stack:
                print(f"栈内容: {[str(item) for item in env.env._builder.stack]}")
        except Exception as e:
            print(f"无法获取表达式状态: {e}")

        # Show current tokens
        current_tokens = [token for token in env.env._tokens if token != env.env._tokens[0]]  # Skip BEG
        print(f"当前token序列: {[str(t) for t in current_tokens]}")

        # Get action masks
        action_masks = env.action_masks()
        available_actions = np.where(action_masks)[0]
        print(f"可用动作数量: {len(available_actions)} / {len(action_masks)}")

        # Check if SEP is available
        sep_available = action_masks[env.env.sep_action] if hasattr(env.env, 'sep_action') else False
        print(f"SEP动作可用: {sep_available}")

        # Simple exploration: just pick a random available action
        if len(available_actions) == 0:
            print("❌ 没有可用动作，结束调试")
            break

        selected_action = np.random.choice(available_actions)
        print(f"🎲 随机选择动作: {selected_action}")

        # Get token for selected action
        try:
            token = env.action(selected_action)
            print(f"🏷️ 选择的token: {token}")
            print(f"📂 token类型: {type(token).__name__}")
        except Exception as e:
            print(f"❌ token转换失败: {e}")
            break

        # Execute action
        print("⚡ 执行动作...")
        next_state, reward, done, truncated, info = env.step(selected_action)
        episode_reward += reward

        print(f"📈 步骤奖励: {reward:.4f}")
        print(f"💵 累积奖励: {episode_reward:.4f}")
        print(f"🏁 episode结束: {done}")

        if done:
            print("\n🎉 Episode完成!")
            print(f"最终token序列: {[str(t) for t in env.env._tokens[1:] if not isinstance(t, SequenceIndicatorToken)]}")
            print(f"总奖励: {episode_reward:.4f}")
            break

    print("\n📋 Episode总结:")
    print(f"总步骤: {step}")
    print(f"最终奖励: {episode_reward:.4f}")
    final_expr = [str(t) for t in env.env._tokens[1:] if not isinstance(t, SequenceIndicatorToken)]
    print(f"最终表达式: {' '.join(final_expr) if final_expr else '空表达式'}")

if __name__ == "__main__":
    debug_single_episode()
