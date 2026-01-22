import sys
import os
import itertools
from collections import defaultdict

# ==========================================
# 0. 加载环境与配置
# ==========================================
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from adapters.field_config import field_config
    from adapters.dic_lol import result_dict as FIELD_META
    
    # 1. 获取环境实际使用的字段 (695个)
    ENV_FIELDS = set(field_config.get_field_names())
    
    # 建立模糊索引 (去掉Feature包装、去掉下划线、忽略大小写)
    # 目的：为了识别那些仅仅是名字写法不同，但实际上存在的字段
    def normalize(name):
        s = name.replace("Feature(", "").replace(")", "").strip()
        s = s.replace("_", "").replace(".", "").lower()
        return s

    ENV_INDEX = {normalize(f) for f in ENV_FIELDS}

except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ==========================================
# 1. 生成全量字典 (1744个)
# ==========================================
def generate_full_dict():
    full_list = []
    for tmpl_key, (tmpl_type, tmpl_options_groups) in FIELD_META.items():
        if not tmpl_options_groups:
            full_list.append(tmpl_key)
            continue
        
        format_str = tmpl_key.replace('_', '{}')
        try:
            # 兼容处理
            clean_groups = []
            for g in tmpl_options_groups:
                if isinstance(g, list): clean_groups.append([str(x) for x in g])
                else: clean_groups.append([str(g)])
            
            for combination in itertools.product(*clean_groups):
                try:
                    full_list.append(format_str.format(*combination))
                except: pass
        except: pass
    return full_list

# ==========================================
# 2. 核心分析逻辑
# ==========================================
def analyze_reduction():
    print("🚀 开始分析字段筛选逻辑...")
    
    # A. 生成总集
    FULL_DICT_LIST = generate_full_dict()
    print(f"📚 字典理论全集: {len(FULL_DICT_LIST)} 个")
    print(f"🌍 环境实际装载: {len(ENV_FIELDS)} 个")
    
    # B. 找出被丢弃的字段 (Rejected)
    # 逻辑：如果字典里的字段，normalize后不在环境里，那就是真被丢了
    kept_count = 0
    rejected_list = []
    
    for field in FULL_DICT_LIST:
        norm = normalize(field)
        if norm in ENV_INDEX:
            kept_count += 1
        else:
            rejected_list.append(field)
            
    print(f"✅ 保留字段 (Kept): {kept_count} 个")
    print(f"🗑️ 被丢弃/未加载 (Dropped): {len(rejected_list)} 个")
    
    if len(rejected_list) == 0:
        print("🎉 奇怪，没有字段被丢弃？那数量应该对得上啊。")
        return

    # C. 分析丢弃规律 (Clustering)
    # 我们按前缀分组，看看哪类字段死伤惨重
    print("\n🔍 [丢弃规律分析] 看看是谁被删了：")
    
    category_stats = defaultdict(lambda: {"total": 0, "dropped": 0, "examples": []})
    
    for f in FULL_DICT_LIST:
        prefix = f.split('.')[0] # 获取 Preload, CS, Slice
        category_stats[prefix]["total"] += 1
        
        norm = normalize(f)
        if norm not in ENV_INDEX:
            category_stats[prefix]["dropped"] += 1
            if len(category_stats[prefix]["examples"]) < 3:
                category_stats[prefix]["examples"].append(f)

    # 打印分组统计
    print("-" * 60)
    print(f"{'类别':<10} | {'总数':<8} | {'丢弃数':<8} | {'丢弃率':<8} | {'丢弃示例'}")
    print("-" * 60)
    
    for prefix, stats in category_stats.items():
        rate = (stats["dropped"] / stats["total"]) * 100
        examples = ", ".join(stats["examples"])
        print(f"{prefix:<10} | {stats['total']:<8} | {stats['dropped']:<8} | {rate:6.1f}% | {examples}...")

    # D. 深度特征分析 (猜测是否过滤了特定后缀)
    # 比如：是不是所有的 'Sell' 都被丢了？或者所有的 'Vol'？
    print("\n🕵️ [深度特征侦探] 关键词命中率分析:")
    keywords = ["Buy", "Sell", "Amt", "Vol", "Cnt", "1min", "5min", "Ret", "Res"]
    
    print(f"{'关键词':<10} | {'被丢弃的字段里包含此词的数量'}")
    for kw in keywords:
        count = sum(1 for f in rejected_list if kw in f)
        if count > 0:
            print(f"{kw:<10} | {count}")

if __name__ == "__main__":
    analyze_reduction()