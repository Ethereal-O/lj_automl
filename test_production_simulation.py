#!/usr/bin/env python3
"""
模拟生产环境的表达式生成过程
验证：只有在满足SEP停止条件之后才可能停止
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
import random

class MockCalculator(AlphaCalculator):
    """模拟生产环境的计算器"""
    def __init__(self):
        self._agent_ref = None
        
    def calc_single_IC_ret(self, expr: Expression) -> float:
        # 模拟IC计算，返回随机值
        return random.uniform(-0.1, 0.1)
    
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return random.uniform(-0.1, 0.1)
    
    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        return random.uniform(-0.1, 0.1)
    
    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        return random.uniform(-0.1, 0.1)

def simulate_production_episode(env, episode_num: int, max_steps: int = 50):
    """模拟一个完整的生产环境episode"""
    print(f"\n🎬 Episode {episode_num}: 模拟生产环境")
    print("=" * 60)
    
    obs, info = env.reset()
    done = False
    step = 0
    actions_taken = []
    
    # 获取env_core引用
    env_core = env.env
    
    # 模拟智能体选择动作的过程
    while not done and step < max_steps:
        step += 1
        print(f"\n📍 Step {step}")
        
        # 获取当前状态信息
        stack_size = len(env_core._builder.stack)
        print(f"  栈状态: {stack_size} 个元素")
        
        if stack_size > 0:
            for i, expr in enumerate(env_core._builder.stack):
                expr_type = env._get_expr_type(expr)
                print(f"    表达式 {i}: {expr_type}")
        
        # 获取动作掩码
        masks = env.action_masks()
        sep_allowed = masks[env_core.unwrapped.sep_action]
        
        print(f"  SEP允许: {sep_allowed}")
        
        # 检查当前是否满足SEP条件
        is_valid_complete = env._is_valid_complete_expression()
        print(f"  满足SEP条件: {is_valid_complete}")
        
        # 模拟智能体选择动作（优先选择SEP，如果允许的话）
        available_actions = []
        
        # 检查SEP是否可用 - 使用 .any() 来处理numpy数组
        if sep_allowed.any():
            available_actions.append(('SEP', env_core.unwrapped.sep_action))
            print("  ✅ SEP可用，智能体选择停止")
            action_name = 'SEP'
            action_idx = env_core.unwrapped.sep_action
        else:
            # 选择其他可用动作
            # 优先选择字段（增加多样性）
            field_actions = []
            op_actions = []
            
            # 收集可用的字段动作
            for i in range(SIZE_FEATURE):
                action_idx = OFFSET_FEATURE - 1 + i
                if masks[action_idx]:
                    field_actions.append(action_idx)
            
            # 收集可用的算子动作
            for i in range(SIZE_OP):
                action_idx = OFFSET_OP - 1 + i
                if masks[action_idx]:
                    op_actions.append(action_idx)
            
            if field_actions:
                action_idx = random.choice(field_actions)
                action_name = f"Field_{action_idx}"
                print(f"  📊 选择字段动作: {action_name}")
            elif op_actions:
                action_idx = random.choice(op_actions)
                action_name = f"Op_{action_idx}"
                print(f"  🔧 选择算子动作: {action_name}")
            else:
                # 没有可用动作，强制停止
                action_idx = env_core.unwrapped.sep_action
                action_name = 'SEP_Force'
                print(f"  ⚠️ 无可用动作，强制停止")
        
        # 执行动作
        obs, reward, done, truncated, info = env.step(action_idx)
        actions_taken.append(action_name)
        
        print(f"  动作: {action_name}")
        print(f"  奖励: {reward:.4f}")
        print(f"  完成: {done}")
        
        # 如果是SEP动作，检查是否真的满足条件
        if action_name == 'SEP' or action_name == 'SEP_Force':
            final_stack_size = len(env_core._builder.stack)
            print(f"  最终栈大小: {final_stack_size}")
            
            if final_stack_size == 1:
                final_expr_type = env._get_expr_type(env_core._builder.stack[0])
                print(f"  最终表达式类型: {final_expr_type}")
                
                if final_expr_type in ['float', 'int']:
                    print("  ✅ 正确：满足SEP条件后停止")
                else:
                    print("  ❌ 错误：类型不匹配却停止了")
            else:
                print("  ❌ 错误：栈中元素数量不为1却停止了")
        
        # 如果没有完成，继续
        if not done:
            continue
    
    print(f"\n📊 Episode {episode_num} 总结:")
    print(f"  动作序列: {' -> '.join(actions_taken)}")
    print(f"  最终奖励: {reward:.4f}")
    print(f"  步数: {step}")
    
    return done, reward, step, actions_taken

def test_production_simulation():
    """测试生产环境模拟"""
    print("🏭 生产环境模拟测试")
    print("=" * 80)
    
    # 创建环境
    pool = AlphaPool(capacity=10, calculator=MockCalculator())
    env_core = AlphaEnvCore(pool=pool, print_expr=True)
    env = AlphaEnvWrapper(env_core)
    
    # 运行多个episode
    num_episodes = 5
    
    for episode in range(1, num_episodes + 1):
        try:
            done, reward, steps, actions = simulate_production_episode(env, episode)
            
            # 分析episode结果
            if 'SEP' in actions or 'SEP_Force' in actions:
                sep_step = len(actions) - 1  # SEP是最后一步
                print(f"  📝 SEP在第 {sep_step} 步执行")
                
                # 检查SEP执行前的状态
                if sep_step > 0:
                    # 重新模拟到SEP前一步
                    obs, info = env.reset()
                    for i in range(sep_step):
                        action_name = actions[i]
                        if action_name.startswith('Field_'):
                            action_idx = int(action_name.split('_')[1])
                        elif action_name.startswith('Op_'):
                            action_idx = int(action_name.split('_')[1])
                        else:
                            continue
                        obs, _, _, _, _ = env.step(action_idx)
                    
                    # 检查SEP前的状态
                    stack_size = len(env_core._builder.stack)
                    sep_allowed = env.action_masks()[env_core.unwrapped.sep_action]
                    is_valid = env._is_valid_complete_expression()
                    
                    print(f"  🔍 SEP前状态: 栈大小={stack_size}, SEP允许={sep_allowed}, 有效={is_valid}")
                    
                    if not is_valid and sep_allowed:
                        print("  ⚠️  警告：SEP允许但不满足条件！")
                    elif is_valid and sep_allowed:
                        print("  ✅ 正确：SEP允许且满足条件")
                    elif not sep_allowed:
                        print("  ✅ 正确：SEP不允许")
            
        except Exception as e:
            print(f"  ❌ Episode {episode} 出错: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("🎯 生产环境模拟测试完成")
    print("📋 验证要点：")
    print("   1. SEP只在满足条件时才允许")
    print("   2. 智能体只在SEP允许时才可能选择停止")
    print("   3. 不满足条件时，智能体被迫继续生成表达式")

if __name__ == "__main__":
    test_production_simulation()
