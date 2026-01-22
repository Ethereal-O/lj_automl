#!/usr/bin/env python3
"""
测试特定字段的填充逻辑
"""

import os
import sys

# 测试特定的字段模板
def test_specific_field():
    """测试 Slice.Price_Trade_ 字段"""

    # 设置展开模式
    os.environ['ALPHAQCM_FIELD_MODE'] = 'expanded'

    # 重新导入
    if 'adapters.field_config' in sys.modules:
        del sys.modules['adapters.field_config']

    from adapters.field_config import FieldConfig

    config = FieldConfig()

    # 查找包含 "Price_Trade" 的字段
    all_fields = config.get_field_names()
    price_trade_fields = [f for f in all_fields if 'Price' in f and 'Trade' in f]

    print("包含 Price 和 Trade 的字段:")
    for field in price_trade_fields[:10]:  # 只显示前10个
        print(f"  {field}")

    # 测试原始模板
    from adapters.dic_lol import result_dict as FIELD_DICT

    target_template = "Slice.Price_Trade_"
    if target_template in FIELD_DICT:
        field_type, options = FIELD_DICT[target_template]
        print(f"\n原始模板: {target_template}")
        print(f"类型: {field_type}")
        print(f"选项: {options}")

        # 手动生成一些组合
        import itertools
        print("\n手动生成的组合示例:")
        for i, combination in enumerate(itertools.product(*options)):
            if i >= 5:  # 只显示前5个
                break

            # 模拟替换
            field_name = target_template
            for option in combination:
                field_name = field_name.replace('_', option, 1)

            print(f"  {combination} -> {field_name}")

if __name__ == "__main__":
    test_specific_field()