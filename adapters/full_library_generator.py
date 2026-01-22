import itertools
from adapters.dic_lol import result_dict as FIELD_META

def get_all_dict_fields():
    """
    严格按照字典定义，生成所有可能的字段名列表 (1744个)。
    用于指导虚拟数据生成器。
    """
    full_names = []
    
    for tmpl_key, (tmpl_type, tmpl_options_groups) in FIELD_META.items():
        # 1. 静态字段
        if not tmpl_options_groups:
            full_names.append(tmpl_key)
            continue

        # 2. 动态字段
        num_slots = tmpl_key.count('_')
        if num_slots == 0:
            full_names.append(tmpl_key)
            continue

        format_str = tmpl_key.replace('_', '{}')
        
        try:
            # 确保所有选项都是列表形式
            clean_groups = []
            for g in tmpl_options_groups:
                if isinstance(g, list):
                    clean_groups.append([str(x) for x in g])
                else:
                    clean_groups.append([str(g)])

            # 笛卡尔积展开
            for combination in itertools.product(*clean_groups):
                try:
                    name = format_str.format(*combination)
                    full_names.append(name)
                except IndexError:
                    pass
        except Exception:
            pass
            
    # 去重并排序，保证顺序一致
    return sorted(list(set(full_names)))

if __name__ == "__main__":
    fields = get_all_dict_fields()
    print(f"✅ 生成了 {len(fields)} 个标准字段名")
    print(f"示例: {fields[:5]}")