#!/usr/bin/env python3
"""
测试字段配置系统
"""

import os
import sys

# 设置环境变量测试不同模式
def test_field_config(mode='expanded'):
    """测试字段配置"""
    os.environ['ALPHAQCM_FIELD_MODE'] = mode

    # 重新导入以使用新环境变量
    if 'adapters.field_config' in sys.modules:
        del sys.modules['adapters.field_config']

    from adapters.field_config import field_config

    print(f"\n=== 测试 {mode} 模式 ===")
    print(f"模式: {field_config.mode}")

    # 获取字段数量
    field_names = field_config.get_field_names()
    print(f"字段数量: {len(field_names)}")

    # 显示前几个字段
    print("前5个字段:")
    for i, name in enumerate(field_names[:5]):
        info = field_config.get_field_info(name)
        print(f"  {name}: {info}")

    # 测试模板字段识别
    template_count = sum(1 for name in field_names if field_config.is_template_field(name))
    print(f"模板字段数量: {template_count}")

    # 测试随机填充
    if mode == 'random' and template_count > 0:
        template_fields = [name for name in field_names if field_config.is_template_field(name)]
        if template_fields:
            test_field = template_fields[0]
            filled = field_config.get_random_filled_field(test_field)
            print(f"随机填充示例: {test_field} -> {filled}")


if __name__ == "__main__":
    # 测试展开模式
    test_field_config('expanded')

    # 测试随机模式
    test_field_config('random')

    print("\n=== 使用说明 ===")
    print("设置环境变量 ALPHAQCM_FIELD_MODE 来控制字段模式:")
    print("  expanded: 使用全展开字段字典（默认）")
    print("  random: 使用模板字段，随机填充")
    print("\n例如: export ALPHAQCM_FIELD_MODE=random")