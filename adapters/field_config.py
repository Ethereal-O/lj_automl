"""
字段配置管理 (大字典模式)
逻辑：
1. 读取 raw 模板字典
2. 遍历每一个模板
3. 将模板的所有可能性全部展开（填充下划线）
4. 将生成的所有干净名字作为 Key，模板的类型作为 Value，存入最终的 self.field_dict
5. Wrapper 直接查这个 self.field_dict，无需任何模糊匹配
"""

import os
import random
import itertools
from typing import Dict, Tuple, List, Any
# 导入原始模板字典
from adapters.dic_lol import result_dict as RAW_TEMPLATE_DICT

class FieldConfig:
    """字段配置管理器"""

    def __init__(self):
        # 模式：expanded (全量 1744) 或 random (随机训练用)
        self.mode = os.environ.get('ALPHAQCM_FIELD_MODE', 'expanded').lower()
        
        # 核心：在这里构建“大字典”
        # 结构：{ "Slice.Cum": ("float", []), "Slice.CumAggOrder": ("float", []), ... }
        self.field_dict = self._build_flat_dictionary()

    def _build_flat_dictionary(self) -> Dict[str, Tuple[str, List]]:
        """
        构建平铺的大字典 (The Big Dictionary)
        """
        flat_dict = {}

        # 遍历原始模板字典
        for tmpl_key, (tmpl_type, tmpl_options_groups) in RAW_TEMPLATE_DICT.items():
            
            # 1. 生成该模板下的所有具体名字列表
            generated_names = []
            
            if not tmpl_options_groups:
                # 情况A: 静态字段 (无下划线，无选项)
                generated_names.append(tmpl_key)
            else:
                # 情况B: 动态字段 (需根据模式填充)
                if self.mode == 'expanded':
                    # 全展开模式：生成所有排列组合
                    generated_names = self._expand_all_variants(tmpl_key, tmpl_options_groups)
                elif self.mode == 'random':
                    # 随机模式：只生成一个随机组合
                    generated_names = [self._generate_one_random(tmpl_key, tmpl_options_groups)]
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

            # 2. 【关键一步】将生成的名字和类型绑定，存入大字典
            # 逻辑：这一批生成的名字，全部继承模板的 type
            for clean_name in generated_names:
                # 存入大字典
                # Key: 干净的名字 (无占位符)
                # Value: (类型, 空列表) -> 既然已经展开，就不再需要 options 了
                flat_dict[clean_name] = (tmpl_type, [])

        return flat_dict

    def _expand_all_variants(self, tmpl_key: str, options_groups: List[List[str]]) -> List[str]:
        """
        工具函数：将一个模板展开成所有可能的干净名字
        """
        variants = []
        
        # 1. 准备格式化字符串
        # 逻辑：把下划线换成 {}，这样 format 时会自动填入
        # 如果填入的是空串 ""，下划线就直接消失了
        num_slots = tmpl_key.count('_')
        if num_slots > 0:
            format_str = tmpl_key.replace('_', '{}')
        else:
            # 防御：没有下划线但有选项，默认追加
            format_str = tmpl_key + "{}" * len(options_groups)

        # 2. 清洗选项 (确保都是字符串列表)
        clean_groups = []
        for g in options_groups:
            if isinstance(g, list):
                clean_groups.append([str(x) for x in g])
            else:
                clean_groups.append([str(g)])

        # 3. 笛卡尔积填充
        for combination in itertools.product(*clean_groups):
            try:
                # 填充！生成干净名字
                clean_name = format_str.format(*combination)
                variants.append(clean_name)
            except Exception:
                pass
                
        return variants

    def _generate_one_random(self, tmpl_key: str, options_groups: List[List[str]]) -> str:
        """
        工具函数：为一个模板生成一个随机的干净名字
        """
        # 1. 随机选择组合
        chosen_comb = []
        for group in options_groups:
            if isinstance(group, list) and group:
                chosen_comb.append(str(random.choice(group)))
            else:
                chosen_comb.append("") # 空选项填空串

        # 2. 格式化
        num_slots = tmpl_key.count('_')
        if num_slots > 0:
            format_str = tmpl_key.replace('_', '{}')
        else:
            format_str = tmpl_key + "{}" * len(chosen_comb)
            
        return format_str.format(*chosen_comb)

    # ----------------------------------------------------
    # 对外接口：Wrapper 只需要调用这些，拿到的都是现成的数据
    # ----------------------------------------------------

    def get_field_names(self) -> List[str]:
        """直接返回大字典的所有 Key"""
        return sorted(list(self.field_dict.keys()))

    def get_field_type(self, clean_name: str) -> str:
        """核心：Wrapper 用这个查类型"""
        # 直接查大字典，不再需要任何猜测
        if clean_name in self.field_dict:
            return self.field_dict[clean_name][0] # 返回 'float'
        return 'float' # 兜底，理论上不应该走到这

    def get_field_info(self, clean_name: str) -> Tuple[str, List[str]]:
        return self.field_dict.get(clean_name, ('unknown', []))

# 全局单例
field_config = FieldConfig()

if __name__ == "__main__":
    # 自检代码
    print(f"当前模式: {field_config.mode}")
    names = field_config.get_field_names()
    print(f"大字典条目数: {len(names)}")
    
    # 验证 Slice.Cum
    test_key = "Slice.Cum"
    if test_key in field_config.field_dict:
        t = field_config.get_field_type(test_key)
        print(f"✅ {test_key} 存在，类型为: {t}")
    else:
        # 如果字典里只有 Slice.CumAggOrder 而没有 Slice.Cum，说明生成逻辑有误(或者字典本身没定义空选项)
        print(f"⚠️ {test_key} 不在字典中 (检查是否字典定义包含空选项)")
        
    # 打印前5个看看样子
    print(f"示例字段: {names[:5]}")