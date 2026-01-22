import numpy as np
import torch
from alphagen.rl.env.wrapper import AlphaEnvWrapper, action2token, OFFSET_OP, OFFSET_FEATURE, OFFSET_SEP
from alphagen.config import OPERATORS

# 模拟环境类，必须继承 gym.Env 以绕过 gymnasium 的检查
import gymnasium as gym
class MockCore(gym.Env):
    def __init__(self):
        super().__init__()
        self._tokens = []
        self._builder = self
        self.stack = [] 
        self.observation_space = gym.spaces.Box(low=0, high=3000, shape=(256,))
        self.action_space = gym.spaces.Discrete(OFFSET_SEP)

def debug_masks():
    print("--- 🛠️ Action Mask 详细条目审计 ---")
    core = MockCore()
    # 尝试实例化你的真实 Wrapper
    try:
        wrapper = AlphaEnvWrapper(core)
    except Exception as e:
        print(f"❌ 实例化 Wrapper 失败: {e}")
        return

    def print_allowed_ops(scenario_name, stack_content):
        core.stack = stack_content
        mask = wrapper.action_masks()
        
        # 提取算子部分的 mask
        op_mask = mask[OFFSET_OP-1 : OFFSET_FEATURE-1]
        allowed_indices = np.where(op_mask)[0]
        
        print(f"\n🚀 场景: {scenario_name} (Stack Size: {len(stack_content)})")
        print(f"允许的算子总数: {len(allowed_indices)}")
        
        if len(allowed_indices) > 0:
            print("具体允许的算子列表:")
            for idx in allowed_indices:
                # 这里的索引转换必须极其精确
                action_idx = idx + OFFSET_OP - 1
                token = action2token(action_idx)
                print(f"  - ID: {action_idx:4d} | Name: {token}")
        else:
            print("  (无允许算子)")

    # 场景 1: 初始状态
    print_allowed_ops("初始状态 (Stack=0)", [])

    # 场景 2: 有一个特征 (模拟一元算子检查)
    class MockExpr: 
        def __init__(self): self.return_type = "float"
        def __str__(self): return "Feature(close)"
    print_allowed_ops("BEG + 字段 (Stack=1)", [MockExpr()])

    # 场景 3: 边界检查
    print(f"\n📊 偏移量定义确认:")
    print(f"OFFSET_OP: {OFFSET_OP} | 第一个算子: {action2token(OFFSET_OP-1)}")
    print(f"OFFSET_FEATURE: {OFFSET_FEATURE} | 第一个特征: {action2token(OFFSET_FEATURE-1)}")
    print(f"OFFSET_SEP: {OFFSET_SEP} | SEP: {action2token(OFFSET_SEP-1)}")

if __name__ == "__main__":
    debug_masks()