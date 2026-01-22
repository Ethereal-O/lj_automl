import sys
import os
import numpy as np
import gymnasium as gym

# 路径适配
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adapters.field_config import field_config
from alphagen.rl.env.wrapper import AlphaEnv, OFFSET_OP, OFFSET_FEATURE, OFFSET_SEP
from alphagen.config import OPERATORS
from adapters.operator_library import OPERATOR_SIGNATURES

# Mock Pool 用于初始化环境
class MockPool:
    def __init__(self):
        self.size = 10
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return None

def find_operator_index(op_name):
    """辅助函数：查找算子对应的动作ID"""
    for i, op in enumerate(OPERATORS):
        # 兼容不同的算子类名定义方式
        name = getattr(op, 'name', op.__class__.__name__)
        if name.lower() == op_name.lower():
            return OFFSET_OP + i
    return None

def find_valid_float_feature_indices(count=3):
    """辅助函数：从大字典里找几个 float 类型的字段索引"""
    names = field_config.get_field_names()
    indices = []
    found_names = []
    
    for i, name in enumerate(names):
        # 只要 float 类型的，保证算子兼容性
        if field_config.get_field_type(name) == 'float':
            indices.append(OFFSET_FEATURE + i)
            found_names.append(name)
            if len(indices) >= count:
                break
    return indices, found_names

def test_complex_generation():
    print("==================================================")
    print("🚀 复杂序列生成与停止逻辑测试")
    print("==================================================")

    # 1. 初始化
    env = AlphaEnv(pool=MockPool())
    obs, info = env.reset()
    
    # 2. 准备素材
    # 找 3 个 float 字段
    feat_indices, feat_names = find_valid_float_feature_indices(3)
    if len(feat_indices) < 3:
        print("❌ 错误：字典里找不到足够的 float 字段，无法测试。")
        return
    
    f1_idx, f2_idx, f3_idx = feat_indices
    print(f"📋 选用测试字段: {feat_names}")

    # 找二元算子（输出必须是float/int，保证满足停止条件）
    binary_op_idx = None
    binary_op_name = "Unknown"
    for i, op in enumerate(OPERATORS):
        if op.n_args() != 2:
            continue
        op_name = getattr(op, 'name', op.__class__.__name__)
        signature = OPERATOR_SIGNATURES.get(op_name)
        if not signature:
            continue
        _, return_type = signature
        if return_type in ["float", "int"]:
            binary_op_idx = OFFSET_OP + i
            binary_op_name = op_name
            break

    if binary_op_idx is None:
        print("❌ 错误：找不到输出为float/int的二元算子，无法测试。")
        return
    print(f"🛠 选用二元算子: {binary_op_name}")

    sep_idx = OFFSET_SEP  # 在你的 wrapper 中，sep_action 应该是 OFFSET_SEP 或 OFFSET_SEP-1，我们直接用 OFFSET_SEP 对应动作空间最后一位
    # 如果 wrapper 是 action_space = SIZE_ALL - SIZE_NULL，那么 SEP 应该是最后一个
    # 你的代码里: action = action_raw + 1 => action == OFFSET_SEP
    # 所以 action_raw (我们要传给 step 的) = OFFSET_SEP - 1
    # 让我们确认一下 OFFSET_SEP 的定义。通常 SEP 是最后一个。
    # 根据你发的 wrapper:
    # mask[OFFSET_SEP - 1] = ...
    # 所以 action_idx 应该是 OFFSET_SEP - 1
    sep_action_id = OFFSET_SEP - 1

    # =========================================================
    # 场景目标：构建 nested 公式: Op(Op(F1, F2), F3)
    # 对应后缀表达式(逆波兰): F1, F2, Op, F3, Op
    # =========================================================

    print("\n---------- 步骤 1: 压入第一个字段 F1 ----------")
    obs, _, _, _, _ = env.step(f1_idx - 1) # 注意 wrapper 里的 action2token 是 action_raw + 1，所以这里要 -1
    # 此时栈: [F1]
    # 期望: 不可停止 (栈大小1，但还没做运算，或者仅仅是一个字段也可以停止？)
    # 通常单独一个字段也是合法公式，但这取决于 valid_complete_expression 的定义
    masks = env.action_masks()
    can_stop = masks[sep_action_id]
    print(f"栈状态: [F1]")
    print(f"允许停止(SEP)? {'✅ 是' if can_stop else '🚫 否'}")

    print("\n---------- 步骤 2: 压入第二个字段 F2 ----------")
    obs, _, _, _, _ = env.step(f2_idx - 1)
    # 此时栈: [F1, F2]
    # 期望: 绝对不可停止 (栈有两个元素，不是单一根节点)
    masks = env.action_masks()
    can_stop = masks[sep_action_id]
    print(f"栈状态: [F1, F2]")
    print(f"允许停止(SEP)? {'❌ 错误 (不应允许)' if can_stop else '✅ 正确 (禁止)'}")
    if can_stop: print("⚠️ 警告：检测到逻辑漏洞，多元素栈允许停止！")

    print(f"\n---------- 步骤 3: 应用算子 {binary_op_name} ----------")
    obs, _, _, _, _ = env.step(binary_op_idx - 1)
    # 此时栈: [Op(F1, F2)]
    # 这是一个完整的 Expression
    # 期望: 可以停止
    masks = env.action_masks()
    can_stop = masks[sep_action_id]
    print(f"栈状态: [{binary_op_name}(F1, F2)]")
    print(f"允许停止(SEP)? {'✅ 正确 (允许)' if can_stop else '❌ 错误 (应允许)'}")

    print("\n---------- 步骤 4: 压入第三个字段 F3 (继续构建) ----------")
    obs, _, _, _, _ = env.step(f3_idx - 1)
    # 此时栈: [Op(F1, F2), F3]
    # 期望: 不可停止
    masks = env.action_masks()
    can_stop = masks[sep_action_id]
    print(f"栈状态: [Result1, F3]")
    print(f"允许停止(SEP)? {'❌ 错误 (不应允许)' if can_stop else '✅ 正确 (禁止)'}")

    print(f"\n---------- 步骤 5: 再次应用算子 {binary_op_name} ----------")
    obs, _, _, _, _ = env.step(binary_op_idx - 1)
    # 此时栈: [Op(Op(F1, F2), F3)]
    # 期望: 可以停止
    masks = env.action_masks()
    can_stop = masks[sep_action_id]
    print(f"栈状态: [{binary_op_name}(Result1, F3)]")
    print(f"允许停止(SEP)? {'✅ 正确 (允许)' if can_stop else '❌ 错误 (应允许)'}")

    print("\n---------- 步骤 6: 执行 SEP (结束生成) ----------")
    if can_stop:
        obs, reward, done, truncated, info = env.step(sep_action_id)
        print(f"执行结果: Done={done}")
        if done:
            print("🎉 序列生成完美结束！")
            # 尝试打印生成的表达式（如果 wrapper 或 env 支持）
            try:
                # 访问 env 内部的 token 列表来重组表达式用于展示
                print("生成的 Token 序列:", [str(t) for t in env.unwrapped._tokens])
            except:
                pass
        else:
            print("❌ 错误：执行 SEP 后环境未返回 done=True")
    else:
        print("❌ 无法执行 SEP，测试在最后一步失败。")

if __name__ == "__main__":
    test_complex_generation()
