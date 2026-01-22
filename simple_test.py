import gymnasium as gym
import numpy as np
import random
from typing import List, Tuple

# ==========================================
# 1. 资源加载 (读取你的配置)
# ==========================================
try:
    # 尝试导入你的真实配置
    from adapters.field_config import field_config
    FIELD_NAMES = field_config.get_field_names()
    from alphagen.config import OPERATORS
    print(f"📚 成功加载配置: {len(FIELD_NAMES)} 个字段, {len(OPERATORS)} 个算子")
except ImportError:
    # 兜底数据 (防止报错无法运行)
    print("⚠️ 未找到配置文件，使用 Mock 数据模式...")
    FIELD_NAMES = [f"close_{i}" for i in range(10)]
    class MockOp:
        def __init__(self, name, n_args): self.name, self._n = name, n_args
        def n_args(self): return self._n
        def __str__(self): return self.name
    OPERATORS = [MockOp("Add", 2), MockOp("Sub", 2), MockOp("Abs", 1), MockOp("Ts_Mean", 2)]

# ==========================================
# 2. 极简环境定义
# ==========================================
class SimpleGenEnv(gym.Env):
    def __init__(self, max_steps=20, max_stack_depth=3):
        super().__init__()
        self.max_steps = max_steps
        self.max_stack_depth = max_stack_depth  # 🔥 新增限制：最大堆栈深度
        
        # --- 动作空间映射 ---
        self.ops = OPERATORS
        self.fields = FIELD_NAMES
        
        # ID 偏移量设计
        # [0...N_OP-1] -> 算子
        # [N_OP...N_OP+N_FIELD-1] -> 字段
        # [最后] -> SEP
        self.offset_op = 0
        self.offset_field = len(self.ops)
        self.offset_sep = self.offset_field + len(self.fields)
        
        self.n_actions = self.offset_sep + 1
        
        # 定义 Gym 空间
        self.action_space = gym.spaces.Discrete(self.n_actions)
        # 观测空间：固定长度的整数数组
        self.observation_space = gym.spaces.Box(
            low=0, high=self.n_actions, shape=(max_steps,), dtype=np.int32
        )
        
        # 内部状态
        self.current_step_count = 0
        self.generated_ids = []   # 记录生成的 token ID 序列
        self.stack_depth = 0      # 核心状态：当前栈里有几个元素
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self.generated_ids = []
        self.stack_depth = 0
        return self._get_obs(), {}

    def step(self, action: int):
        # 1. 记录动作
        self.generated_ids.append(action)
        self.current_step_count += 1
        
        truncated = False
        terminated = False
        reward = 0.0
        info = {}

        # 2. 解析动作并更新堆栈状态
        if action < self.offset_field:
            # ---> 算子 (Operator)
            op = self.ops[action]
            n_args = op.n_args()
            # RPN逻辑：消耗 n 个参数，生成 1 个结果
            # 净变化 = 1 - n
            self.stack_depth -= (n_args - 1)
            
        elif action < self.offset_sep:
            # ---> 字段 (Feature)
            # 压栈，深度 +1
            self.stack_depth += 1
            
        elif action == self.offset_sep:
            # ---> SEP (停止)
            terminated = True
            reward = 1.0 # 成功生成奖励
            
        else:
            # 异常情况
            truncated = True
            reward = -1.0
            
        # 3. 长度检查 (超过20步强制截断)
        if self.current_step_count >= self.max_steps:
            truncated = True
            if not terminated:
                info['reason'] = 'max_steps_reached'
        
        # 4. 返回
        return self._get_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        🔥 核心逻辑：决定哪些动作现在能选
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        
        # --- Rule 1: 字段 (Feature) ---
        # 只有在栈深度还没满的时候，才允许加字段
        # 这就是你要的：假如已经有3个部分了(stack_depth >= 3)，这里就会是 False
        if self.stack_depth < self.max_stack_depth:
            mask[self.offset_field : self.offset_sep] = True
            
        # --- Rule 2: 算子 (Operator) ---
        # 只有栈里的数足够算子吃的时候，才允许选
        # 例如：Add 需要 2 个数，只有 stack_depth >= 2 才能选 Add
        for i, op in enumerate(self.ops):
            if self.stack_depth >= op.n_args():
                mask[i] = True
                
        # --- Rule 3: 停止符 (SEP) ---
        # 只有栈里剩 1 个完整结果，且不是第 0 步时，才允许停
        if self.stack_depth == 1 and self.current_step_count > 0:
            mask[self.offset_sep] = True
            
        return mask

    def _get_obs(self):
        # 自动 Padding 到固定长度，保证 Agent 不报错
        obs = np.array(self.generated_ids, dtype=np.int32)
        if len(obs) < self.max_steps:
            padding = np.zeros(self.max_steps - len(obs), dtype=np.int32)
            obs = np.concatenate([obs, padding])
        return obs[:self.max_steps]

    def decode_expression(self):
        """Debug 用：把 ID 翻译成人话"""
        res = []
        for a in self.generated_ids:
            if a < self.offset_field:
                res.append(f"Op({self.ops[a]})")
            elif a < self.offset_sep:
                res.append(f"Field({self.fields[a - self.offset_field]})")
            elif a == self.offset_sep:
                res.append("SEP")
        return " -> ".join(res)

# ==========================================
# 3. 验证测试
# ==========================================
if __name__ == "__main__":
    print("\n🚀 启动纯净版生成器 (Max Depth=3 测试)")
    env = SimpleGenEnv(max_steps=20, max_stack_depth=2)
    
    for i in range(5):
        print(f"\n🎬 Episode {i+1}:")
        obs, _ = env.reset()
        done = False
        
        while not done:
            # 1. 获取 Mask
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) == 0:
                print("❌ 死局 (无合法动作)")
                break
            
            # 2. 随机采样
            action = np.random.choice(valid_indices)
            
            # 3. 执行
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
            # 打印当前状态
            stack_status = "🟥 满" if env.stack_depth >= 3 else f"{env.stack_depth}"
            print(f"  Step {env.current_step_count:2d} | 栈深: {stack_status} | 动作ID: {action}")

        print(f"  📝 结果: {env.decode_expression()}")
        if trunc: print("  ⚠️  触发截断 (过长)")
        if term:  print("  ✅  成功生成")