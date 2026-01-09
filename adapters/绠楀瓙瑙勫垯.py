'''
# ============================================
# 基础类型系统（语法必须严格匹配）
# ============================================

BASE_TYPES = {
    "float", "int", "bool", 
    "vector", "num_vector", "bool_vector", "index_vector", "mask_vector", "prob_vector",
    "tensor", "field", "factor",
    "const_float", "const_int", "const_bool", "const_vector",
    "expr", "same_as_input", "expr_return_type", "any"
}

# ============================================
# 按取值范围分组（用于RL奖励设计）
# ============================================

VALUE_RANGE_GROUPS = {
    # int分组
    "small_int": "int",      # 小整数：1-100（窗口大小、长度、预留大小等）
    "medium_int": "int",     # 中等整数：1-1000（索引、偏移量等）
    "large_int": "int",      # 大整数：1000+（时间戳、大索引等）
    "group_id_int": "int",   # 分组ID：0-50（分组、桶ID等）
    "precision_int": "int",  # 精度：-5到5（Round的精度参数）
    
    # float分组
    "pct_float": "float",    # 百分比：0-100
    "small_float": "float",  # 小浮点：0-10（比率、阈值等）
    "medium_float": "float", # 中等浮点：任意范围的一般数值
    "price_like_float": "float", # 价格类：通常10-1000
    
    # vector分组（按元素类型）
    "num_vector": "vector",  # 数值向量
    "bool_vector": "vector", # 布尔向量
    "index_vector": "vector", # 索引向量（元素为整数）
    "mask_vector": "vector", # 掩码向量
    "prob_vector": "vector", # 概率向量
}
'''
# ============================================
# 完整类型系统（带继承层次）
# ============================================

# 类型继承关系定义
TYPE_HIERARCHY = {
    # 顶层类型（最通用）
    "any": {"parent": None, "desc": "任何类型"},
    
    # 表达式类型
    "expr": {"parent": "any", "desc": "表达式"},
    "expr_return_type": {"parent": "expr", "desc": "表达式返回值类型"},
    "same_as_input": {"parent": "expr", "desc": "与输入相同类型"},
    
    # 标量类型
    "float": {"parent": "expr", "desc": "浮点数"},
    "int": {"parent": "float", "desc": "整数（浮点数子类）"},
    "bool": {"parent": "int", "desc": "布尔值（整数子类）"},
    
    # 常量标量类型（对应非常量类型的子类）
    "const_float": {"parent": "float", "desc": "常量浮点数"},
    "const_int": {"parent": "int", "desc": "常量整数"},
    "const_bool": {"parent": "bool", "desc": "常量布尔值"},
    
    # 复合类型
    "vector": {"parent": "expr", "desc": "向量/数组"},
    "tensor": {"parent": "expr", "desc": "张量/矩阵"},
    "field": {"parent": "expr", "desc": "基础字段"},
    "factor": {"parent": "expr", "desc": "因子引用"},
    
    # 常量复合类型
    "const_vector": {"parent": "vector", "desc": "常量向量"},
    
    # 向量子类型
    "num_vector": {"parent": "vector", "desc": "数值向量"},
    "bool_vector": {"parent": "vector", "desc": "布尔向量"},
    "index_vector": {"parent": "vector", "desc": "索引向量"},
    "mask_vector": {"parent": "vector", "desc": "掩码向量"},
    "prob_vector": {"parent": "vector", "desc": "概率向量"},
}

# 逆向查找：子类型到父类型的映射
def get_all_parents(type_name):
    """获取类型的所有父类（包括间接父类）"""
    parents = set()
    current = type_name
    
    while current in TYPE_HIERARCHY:
        parent = TYPE_HIERARCHY[current]["parent"]
        if parent:
            parents.add(parent)
            current = parent
        else:
            break
    
    return parents

def is_subtype(child_type, parent_type):
    """检查child_type是否是parent_type的子类型（包括间接继承）"""
    if child_type == parent_type:
        return True
    
    # 检查直接或间接父类
    parents = get_all_parents(child_type)
    return parent_type in parents

def get_type_compatibility(expected_type, actual_type):
    """
    获取类型兼容性分数
    1.0: 完全匹配或实际类型是期望类型的子类
    0.0-0.9: 不同程度的兼容
    0.0: 完全不兼容
    """
    # 特殊情况
    if expected_type == "any":
        return 1.0
    
    if expected_type == "same_as_input":
        # 需要在调用处特殊处理
        return 1.0
    
    if expected_type == "expr":
        # expr可以匹配任何表达式类型
        return 1.0 if actual_type in ["float", "int", "bool", "vector", "tensor", 
                                      "field", "factor", "const_float", "const_int",
                                      "const_bool", "const_vector", "num_vector",
                                      "bool_vector", "index_vector", "mask_vector",
                                      "prob_vector"] else 0.0
    
    # expr_return_type比较特殊，需要根据上下文判断
    if expected_type == "expr_return_type":
        return 0.8  # 中等惩罚，因为不够明确
    
    # 完全匹配
    if expected_type == actual_type:
        return 1.0
    
    # 检查继承关系
    if is_subtype(actual_type, expected_type):
        # 实际类型是期望类型的子类：完全兼容
        return 1.0
    
    # 检查是否是常量类型的关系
    if expected_type.startswith("const_"):
        # 期望常量类型，实际是非常量对应类型
        base_expected = expected_type[6:]  # 去掉"const_"
        if actual_type == base_expected:
            return 0.8  # 轻微惩罚：常量vs非常量
    
    # 检查是否是vector基类与子类的关系
    if expected_type == "vector":
        if actual_type in ["num_vector", "bool_vector", "index_vector", 
                          "mask_vector", "prob_vector", "const_vector"]:
            return 1.0  # vector接受任何vector子类
    
    # vector子类期望，但给的是vector基类
    elif expected_type in ["num_vector", "bool_vector", "index_vector", 
                          "mask_vector", "prob_vector"]:
        if actual_type == "vector":
            return 0.9  # 轻微惩罚：不够具体
        elif actual_type == "const_vector":
            return 0.7  # 常量向量可能不是期望的子类型
    
    # 标量类型之间的特殊兼容性 - 扩展完整矩阵
    scalar_compat_matrix = {
        # ===== 基础类型转换 =====
        # float作为目标类型
        ("float", "float"): 1.0,           # 完全匹配
        ("float", "int"): 0.8,             # int->float：安全转换
        ("float", "bool"): 0.0,            # bool->float：不推荐
        ("float", "const_float"): 0.9,     # const_float->float：常量优势
        ("float", "const_int"): 0.7,       # const_int->float：常量+转换
        ("float", "const_bool"): 0.0,      # const_bool->float：不推荐

        # int作为目标类型
        ("int", "int"): 1.0,               # 完全匹配
        ("int", "float"): 0.0,             # float->int：可能截断
        ("int", "bool"): 0.0,              # bool->int：不推荐
        ("int", "const_int"): 0.9,         # const_int->int：常量优势
        ("int", "const_float"): 0.0,       # const_float->int：截断+常量
        ("int", "const_bool"): 0.0,        # const_bool->int：不推荐

        # bool作为目标类型
        ("bool", "bool"): 1.0,             # 完全匹配
        ("bool", "int"): 0.0,              # int->bool：不推荐
        ("bool", "float"): 0.0,            # float->bool：不推荐
        ("bool", "const_bool"): 0.9,       # const_bool->bool：常量优势
        ("bool", "const_int"): 0.0,        # const_int->bool：不推荐
        ("bool", "const_float"): 0.0,      # const_float->bool：不推荐

        # ===== 常量类型作为目标 =====
        # const_float作为目标
        ("const_float", "const_float"): 1.0,  # 完全匹配
        ("const_float", "float"): 0.0,        # 需要常量但给了变量
        ("const_float", "int"): 0.0,          # 类型不匹配
        ("const_float", "bool"): 0.0,         # 类型不匹配
        ("const_float", "const_int"): 0.0,    # 类型不匹配
        ("const_float", "const_bool"): 0.0,   # 类型不匹配

        # const_int作为目标
        ("const_int", "const_int"): 1.0,      # 完全匹配
        ("const_int", "int"): 0.0,            # 需要常量但给了变量
        ("const_int", "float"): 0.0,          # 类型不匹配+截断
        ("const_int", "bool"): 0.0,           # 类型不匹配
        ("const_int", "const_float"): 0.0,    # 类型不匹配
        ("const_int", "const_bool"): 0.0,     # 类型不匹配

        # const_bool作为目标
        ("const_bool", "const_bool"): 1.0,    # 完全匹配
        ("const_bool", "bool"): 0.0,          # 需要常量但给了变量
        ("const_bool", "int"): 0.0,           # 类型不匹配
        ("const_bool", "float"): 0.0,         # 类型不匹配
        ("const_bool", "const_int"): 0.0,     # 类型不匹配
        ("const_bool", "const_float"): 0.0,   # 类型不匹配

        # ===== 向量类型兼容性 =====
        # num_vector作为目标
        ("num_vector", "num_vector"): 1.0,
        ("num_vector", "vector"): 0.9,        # 基类到子类
        ("num_vector", "bool_vector"): 0.0,   # 不同子类
        ("num_vector", "index_vector"): 0.7,  # 索引向量可能兼容数值
        ("num_vector", "mask_vector"): 0.0,   # 掩码不兼容
        ("num_vector", "prob_vector"): 0.8,   # 概率向量通常是数值

        # bool_vector作为目标
        ("bool_vector", "bool_vector"): 1.0,
        ("bool_vector", "vector"): 0.8,       # 基类到子类
        ("bool_vector", "num_vector"): 0.0,   # 数值向量到布尔（不安全）
        ("bool_vector", "index_vector"): 0.0, # 索引不兼容布尔
        ("bool_vector", "mask_vector"): 0.9,  # 掩码通常是布尔
        ("bool_vector", "prob_vector"): 0.6,  # 概率到布尔（阈值转换）

        # index_vector作为目标
        ("index_vector", "index_vector"): 1.0,
        ("index_vector", "vector"): 0.7,      # 基类兼容性
        ("index_vector", "num_vector"): 0.8,  # 数值向量转索引（取整）
        ("index_vector", "bool_vector"): 0.0, # 布尔到索引无意义
        ("index_vector", "mask_vector"): 0.6, # 掩码到索引（位置编码）
        ("index_vector", "prob_vector"): 0.0, # 概率到索引无意义

        # mask_vector作为目标
        ("mask_vector", "mask_vector"): 1.0,
        ("mask_vector", "vector"): 0.8,       # 基类兼容性
        ("mask_vector", "bool_vector"): 0.9,  # 布尔向量到掩码
        ("mask_vector", "num_vector"): 0.5,   # 数值到掩码（阈值）
        ("mask_vector", "index_vector"): 0.4, # 索引到掩码（存在性）
        ("mask_vector", "prob_vector"): 0.7,  # 概率到掩码（阈值）

        # prob_vector作为目标
        ("prob_vector", "prob_vector"): 1.0,
        ("prob_vector", "vector"): 0.8,       # 基类兼容性
        ("prob_vector", "num_vector"): 0.9,   # 数值到概率（归一化）
        ("prob_vector", "bool_vector"): 0.0,  # 布尔到概率无意义
        ("prob_vector", "index_vector"): 0.0, # 索引到概率无意义
        ("prob_vector", "mask_vector"): 0.6,  # 掩码到概率（0/1转换）

        # vector作为目标（接受任何子类）
        ("vector", "vector"): 1.0,
        ("vector", "num_vector"): 1.0,        # 子类到基类
        ("vector", "bool_vector"): 1.0,
        ("vector", "index_vector"): 1.0,
        ("vector", "mask_vector"): 1.0,
        ("vector", "prob_vector"): 1.0,
        ("vector", "const_vector"): 0.9,      # 常量向量

        # ===== 常量向量类型 =====
        ("const_vector", "const_vector"): 1.0,
        ("const_vector", "vector"): 0.0,       # 需要常量但给了变量
        ("const_vector", "num_vector"): 0.0,
        ("const_vector", "bool_vector"): 0.0,
        ("const_vector", "index_vector"): 0.0,
        ("const_vector", "mask_vector"): 0.0,
        ("const_vector", "prob_vector"): 0.0,

        # ===== 特殊类型转换 =====
        # factor类型（因子引用）
        ("factor", "factor"): 1.0,
        ("factor", "expr"): 0.7,              # 表达式到因子引用
        ("factor", "float"): 0.0,             # 数值不能直接作为因子引用
        ("factor", "int"): 0.0,
        ("factor", "bool"): 0.0,

        # field类型（基础字段）
        ("field", "field"): 1.0,
        ("field", "expr"): 0.8,               # 表达式可能包含字段
        ("field", "float"): 0.0,              # 数值不是字段
        ("field", "int"): 0.0,
        ("field", "bool"): 0.0,

        # tensor类型
        ("tensor", "tensor"): 1.0,
        ("tensor", "vector"): 0.6,            # 向量到张量（升维）
        ("tensor", "float"): 0.0,             # 标量到张量无意义
        ("tensor", "int"): 0.0,
        ("tensor", "bool"): 0.0,
    }
    
    key = (expected_type, actual_type)
    if key in scalar_compat_matrix:
        return scalar_compat_matrix[key]
    
    # 检查是否是同一继承链上的类型
    expected_parents = get_all_parents(expected_type)
    actual_parents = get_all_parents(actual_type)
    
    # 找最近的共同祖先
    common_parents = expected_parents.intersection(actual_parents)
    if common_parents:
        # 有共同祖先，但不是继承关系
        if "float" in common_parents:
            return 0.6  # 都在数值类型继承链上
        elif "expr" in common_parents:
            return 0.4  # 都是表达式但类型不同
    
    return 0.0  # 完全不兼容

# ============================================
# 类型验证和评分工具
# ============================================

# ============================================
# 类型推断辅助函数
# ============================================

def infer_return_type(operator, input_types):
    """
    根据算子名和输入类型推断返回值类型
    """
    if operator not in OPERATOR_SIGNATURES:
        return None
    
    arg_specs, return_type_spec = OPERATOR_SIGNATURES[operator]
    
    if return_type_spec == "same_as_input":
        # 对于same_as_input，通常返回第一个参数的类型
        return input_types[0] if input_types else None
    
    return return_type_spec

def get_operator_compatibility_score(operator, input_types):
    """
    计算算子与输入类型的整体兼容性分数
    """
    if operator not in OPERATOR_SIGNATURES:
        return 0.0
    
    arg_specs, _ = OPERATOR_SIGNATURES[operator]
    
    # 检查参数数量
    if len(input_types) != len(arg_specs):
        return 0.0
    
    # 计算每个参数的兼容性
    scores = []
    for expected, actual in zip(arg_specs, input_types):
        score = get_type_compatibility(expected, actual)
        scores.append(score)
    
    # 返回平均兼容性分数
    return sum(scores) / len(scores) if scores else 0.0
# ============================================
# 完整算子签名字典（包含细化类型）
# ============================================

OPERATOR_SIGNATURES = {
    # === 历史数据算子 ===
    "HistLastDailyField": (["field"], "field"),
    "HistDailyField": (["field", "const_int", "const_int"], "field"),
    "HistDailyFieldVec": (["field", "const_int"], "vector"),
    "HistLastSliceFieldVec": (["field", "const_int"], "vector"),
    "HistTdVec": (["field", "const_int"], "vector"),
    "HistCDSIVec": (["field", "const_int", "const_int", "const_int"], "vector"),
    "HistSliceFieldTens": (["field", "const_int"], "tensor"),
    "HistTdTens": (["field", "const_int", "const_int"], "tensor"),
    "HistTdMedianVec": (["field", "const_int", "const_int"], "vector"),
    "HistTdMedianTens": (["field", "const_int", "const_int", "const_int"], "tensor"),

    # === 逻辑运算 ===
    "Less": (["float", "float"], "bool"),
    "Greater": (["float", "float"], "bool"),
    "Equal": (["float", "float"], "bool"),
    "NotLess": (["float", "float"], "bool"),
    "NotGreater": (["float", "float"], "bool"),
    "NotEqual": (["float", "float"], "bool"),
    "And": (["bool", "bool"], "bool"),
    "Or": (["bool", "bool"], "bool"),
    "Xor": (["bool", "bool"], "bool"),
    "Not": (["bool"], "bool"),
    "IfElse": (["bool", "expr", "expr"], "expr_return_type"),
    "If": (["bool", "expr"], "expr_return_type"),
    "IsHist": ([], "bool"),
    "IsToday": ([], "bool"),

    # === 调试算子 ===
    "Print": (["any"], "same_as_input"),
    "Pause": (["any", "float"], "same_as_input"),

    # === 标量运算 ===
    "Neg": (["float"], "float"),
    "Inv": (["float"], "float"),
    "Add": (["float", "float"], "float"),
    "Sub": (["float", "float"], "float"),
    "Mul": (["float", "float"], "float"),
    "Div": (["float", "float"], "float"),
    "Max": (["float", "float"], "float"),
    "Min": (["float", "float"], "float"),
    "Abs": (["float"], "float"),
    "Log": (["float"], "float"),
    "Exp": (["float"], "float"),
    "Sqrt": (["float"], "float"),
    "Pow": (["float", "float"], "float"),
    "Ret": (["float", "float"], "float"),
    "LogRet": (["float", "float"], "float"),
    "AbsRet": (["float", "float"], "float"),
    "Imbl": (["float", "float"], "float"),
    "Zero": (["float", "float"], "float"),
    "TR": (["float", "float", "float"], "float"),
    "Floor": (["float"], "int"),
    "Ceil": (["float"], "int"),
    "Round": (["float", "int"], "int"),
    "BRound": (["float", "int"], "int"),
    "Mod": (["int", "int"], "int"),
    "Int": (["float"], "int"),
    "Dec": (["float"], "float"),
    "Bool": (["float"], "bool"),
    "Fill": (["float", "float", "float"], "float"),
    "FillByFlag": (["float", "bool", "float"], "float"),
    "FillNan": (["float", "float"], "float"),
    "FillInf": (["float", "float"], "float"),

    # === 向量运算（细化类型）===
    "VecNew": (["int"], "num_vector"),
    "VecLen": (["vector"], "float"),
    "VecInplacePush": (["num_vector", "float"], "num_vector"),
    "VecInplacePushVec": (["vector", "vector"], "vector"),
    "VecInplaceSort": (["num_vector"], "num_vector"),
    "VecInplaceSortByVec": (["vector", "index_vector"], "vector"),
    "VecPush": (["num_vector", "float"], "num_vector"),
    "VecPushVec": (["vector", "vector"], "vector"),
    "VecSort": (["num_vector"], "num_vector"),
    "VecInplacePushSortKV": (["vector", "float", "float"], "vector"),
    "VecSortByVec": (["vector", "index_vector"], "vector"),
    "VecSortIndex": (["num_vector"], "index_vector"),
    "VecPctFrontPoint": (["num_vector", "float"], "float"),
    "VecPctBackPoint": (["num_vector", "float"], "float"),
    "VecPctFrontSubvec": (["vector", "float"], "vector"),
    "VecPctBackSubvec": (["vector", "float"], "vector"),
    "VecPctSubvec": (["vector", "float", "float"], "vector"),
    "VecMean": (["num_vector"], "float"),
    "VecSum": (["num_vector"], "float"),
    "VecProduct": (["num_vector"], "float"),
    "VecMidIdxValue": (["vector"], "same_as_input"),
    "VecMedian": (["num_vector"], "float"),
    "VecStd": (["num_vector"], "float"),
    "VecSkew": (["num_vector"], "float"),
    "VecKurt": (["num_vector"], "float"),
    "VecArgmin": (["num_vector"], "float"),
    "VecArgmax": (["num_vector"], "float"),
    "VecMultiArgmin": (["num_vector"], "index_vector"),
    "VecMultiArgmax": (["num_vector"], "index_vector"),
    "VecArgminZig": (["num_vector"], "float"),
    "VecArgmaxZig": (["num_vector"], "float"),
    "VecArgminmaxZig": (["num_vector"], "float"),
    "VecValue": (["vector", "float"], "same_as_input"),
    "VecMultiValue": (["vector", "index_vector"], "vector"),
    "VecBackSubvec": (["vector", "float"], "vector"),
    "VecMin": (["num_vector"], "float"),
    "VecMax": (["num_vector"], "float"),
    "VecMinZig": (["num_vector"], "float"),
    "VecMaxZig": (["num_vector"], "float"),
    "VecMinmaxZig": (["num_vector"], "float"),
    "VecCumsumVec": (["num_vector", "float"], "num_vector"),
    "VecPeakNum": (["num_vector"], "float"),
    "VecValleyNum": (["num_vector"], "float"),
    "VecPeakNumZig": (["num_vector"], "float"),
    "VecValleyNumZig": (["num_vector"], "float"),
    "VecWaveUpMagZigVec": (["num_vector"], "num_vector"),
    "VecWaveDownMagZigVec": (["num_vector"], "num_vector"),
    "VecWaveUpLenZigVec": (["num_vector"], "num_vector"),
    "VecWaveDownLenZigVec": (["num_vector"], "num_vector"),
    "VecInv": (["num_vector"], "num_vector"),
    "VecLog": (["num_vector"], "num_vector"),
    "VecAdd": (["num_vector", "num_vector"], "num_vector"),
    "VecSub": (["num_vector", "num_vector"], "num_vector"),
    "VecMul": (["num_vector", "num_vector"], "num_vector"),
    "VecDiv": (["num_vector", "num_vector"], "num_vector"),
    "VecBCAdd": (["num_vector", "float"], "num_vector"),
    "VecBCSub": (["num_vector", "float"], "num_vector"),
    "VecBCMul": (["num_vector", "float"], "num_vector"),
    "VecBCDiv": (["num_vector", "float"], "num_vector"),
    "VecRet": (["num_vector", "num_vector"], "num_vector"),
    "VecLogRet": (["num_vector", "num_vector"], "num_vector"),
    "VecRollingMeanVec": (["num_vector", "int"], "num_vector"),
    "VecSlope": (["num_vector"], "float"),
    "VecNormalize": (["num_vector"], "num_vector"),
    "VecShannonEntropy": (["prob_vector"], "float"),
    "VecCrossEntropy": (["prob_vector", "prob_vector"], "float"),
    "VecKLDivergence": (["prob_vector", "prob_vector"], "float"),
    "VecCorr": (["num_vector", "num_vector"], "float"),
    "VecCov": (["num_vector", "num_vector"], "float"),
    "VecRank": (["num_vector"], "num_vector"),
    "VecLessIndex": (["num_vector", "float"], "index_vector"),
    "VecGreaterIndex": (["num_vector", "float"], "index_vector"),
    "VecOneHot": (["index_vector", "int"], "bool_vector"),
    "VecRangeMask": (["num_vector", "const_float", "const_float", "float", "float"], "mask_vector"),
    "VecRangeFilter": (["num_vector", "const_float", "const_float"], "num_vector"),
    "VecMaskFilter": (["vector", "mask_vector", "bool"], "vector"),
    "VecSpreadMask": (["mask_vector", "const_int", "const_int", "const_int"], "mask_vector"),
    "VecQuadReg": (["num_vector", "num_vector"], "num_vector"),
    "VecZscore": (["num_vector"], "num_vector"),
    "VecOneHotMask": (["index_vector", "int", "float", "float"], "mask_vector"),
    "VecSequence": (["int", "int", "int"], "num_vector"),
    "VecExpDecaySeq": (["int", "int"], "num_vector"),
    "VecReverse": (["vector"], "vector"),
    "VecNegative": (["num_vector"], "num_vector"),
    "VecStepSubvec": (["vector", "int", "int", "int"], "vector"),
    "VecValueIdx": (["num_vector", "float"], "index_vector"),
    "VecFwdFill": (["vector", "float"], "vector"),
    "VecBwdFill": (["vector", "float"], "vector"),
    "VecFill": (["vector", "float", "float"], "vector"),
    "VecFillByVec": (["vector", "float", "vector"], "vector"),
    "VecMask": (["vector", "mask_vector", "float", "bool"], "vector"),
    "VecMaskByVec": (["vector", "mask_vector", "vector", "bool"], "vector"),
    "VecRankRaw": (["num_vector", "bool"], "index_vector"),
    "VecDropValues": (["vector", "vector"], "vector"),
    "VecKeepValues": (["vector", "vector"], "vector"),
    "VecInValues": (["vector", "vector", "bool"], "bool_vector"),
    "VecBool": (["vector"], "bool_vector"),
    "VecNot": (["bool_vector"], "bool_vector"),
    "VecBCEqual": (["vector", "float", "bool"], "bool_vector"),
    "VecBCGreater": (["num_vector", "float"], "bool_vector"),
    "VecBCLess": (["num_vector", "float"], "bool_vector"),
    "VecBCNotGreater": (["num_vector", "float"], "bool_vector"),
    "VecBCNotLess": (["num_vector", "float"], "bool_vector"),
    "VecEqual": (["vector", "vector", "bool"], "bool_vector"),
    "VecGreater": (["num_vector", "num_vector"], "bool_vector"),
    "VecLess": (["num_vector", "num_vector"], "bool_vector"),
    "VecNotGreater": (["num_vector", "num_vector"], "bool_vector"),
    "VecNotLess": (["num_vector", "num_vector"], "bool_vector"),
    "VecAny": (["bool_vector"], "bool"),
    "VecAll": (["bool_vector"], "bool"),
    "VecOr": (["bool_vector", "bool_vector"], "bool_vector"),
    "VecXor": (["bool_vector", "bool_vector"], "bool_vector"),
    "VecAnd": (["bool_vector", "bool_vector"], "bool_vector"),
    "VecPeak": (["num_vector", "float", "float", "float"], "num_vector"),
    "VecPeakIdx": (["num_vector", "float", "float", "float"], "index_vector"),
    "VecPeakCnt": (["num_vector", "float", "float", "float"], "int"),
    "VecValley": (["num_vector", "float", "float", "float"], "num_vector"),
    "VecValleyIdx": (["num_vector", "float", "float", "float"], "index_vector"),
    "VecValleyCnt": (["num_vector", "float", "float", "float"], "int"),

    # === 矩阵运算 ===
    "TensRowVecRef": (["tensor", "float"], "vector"),
    "TensRowVec": (["tensor", "float"], "vector"),
    "TensAdd": (["tensor", "tensor"], "tensor"),
    "TensSub": (["tensor", "tensor"], "tensor"),
    "TensMul": (["tensor", "tensor"], "tensor"),
    "TensDiv": (["tensor", "tensor"], "tensor"),
    "TensRowMeanVec": (["tensor"], "vector"),
    "TensRowMedianVec": (["tensor"], "vector"),
    "TensMean": (["tensor"], "float"),
    "TensMedian": (["tensor"], "float"),
    "TensTranspose": (["tensor"], "tensor"),
    "TensFlatten": (["tensor", "int"], "vector"),

    # === 时序运算 ===
    "TsSum": (["float", "const_int", "const_bool"], "float"),
    "TsMean": (["float", "const_int", "const_bool"], "float"),
    "TsMedian": (["float", "const_int", "const_bool"], "float"),
    "TsArgmax": (["float", "const_int", "const_bool"], "float"),
    "TsArgmin": (["float", "const_int", "const_bool"], "float"),
    "TsMax": (["float", "const_int", "const_bool"], "float"),
    "TsMin": (["float", "const_int", "const_bool"], "float"),
    "TsShift": (["float", "const_int", "const_bool"], "float"),
    "TsDiff": (["float", "const_int", "const_bool"], "float"),
    "TsRet": (["float", "const_int", "const_bool"], "float"),
    "TsLogRet": (["float", "const_int", "const_bool"], "float"),
    "TsAbsRet": (["float", "const_int", "const_bool"], "float"),
    "TsVar": (["float", "const_int", "const_bool"], "float"),
    "TsStd": (["float", "const_int", "const_bool"], "float"),
    "TsSkew": (["float", "const_int", "const_bool"], "float"),
    "TsKurt": (["float", "const_int", "const_bool"], "float"),
    "TsEma": (["float", "const_int", "const_bool"], "float"),
    "TsZscore": (["float", "const_int", "const_bool"], "float"),
    "TsSlope": (["float", "const_int", "const_bool"], "float"),
    "TsRank": (["float", "const_int", "const_bool"], "float"),
    "TsCorr": (["float", "float", "const_int", "const_bool"], "float"),
    "TsCov": (["float", "float", "const_int", "const_bool"], "float"),
    "TsObv": (["float", "float", "const_int", "const_bool"], "float"),
    "TsQuantile": (["float", "float", "const_int", "const_bool"], "float"),
    "TsWinsorize": (["float", "float", "const_int", "const_bool"], "float"),
    "TsMad": (["float", "const_int", "const_bool"], "float"),
    "TsFill": (["float", "float", "const_int", "const_bool"], "float"),
    "TsWeightMean": (["float", "float", "const_int", "const_bool"], "float"),
    "TsWeightCenter": (["float", "const_int", "const_bool"], "float"),
    "TsWindow": (["float", "const_int", "const_bool"], "vector"),
    "TsQuadReg": (["float", "float", "const_int", "const_bool"], "vector"),
    "TsHHI": (["float", "const_int", "const_bool"], "float"),
    "TsGini": (["float", "const_int", "const_bool"], "float"),
    "TsEntropy": (["float", "const_int", "const_bool"], "float"),
    "TsFFT": (["float", "const_int", "const_bool"], "vector"),
    "TsATR": (["float", "float", "float", "const_int", "const_bool"], "float"),
    "TsRankRaw": (["float", "const_int", "const_bool"], "int"),
    "TsFixedTimeWindow": (["float", "int", "const_int", "const_int", "const_bool"], "vector"),

    # === 截面运算 ===
    "CsVec": (["float", "int"], "vector"),
    "CsSum": (["float", "int"], "float"),
    "CsMean": (["float", "int"], "float"),
    "CsMedian": (["float", "int"], "float"),
    "CsVar": (["float", "int"], "float"),
    "CsStd": (["float", "int"], "float"),
    "CsRank": (["float", "int"], "float"),
    "CsZscore": (["float", "int"], "float"),
    "CsWinsorize": (["float", "const_float", "int"], "float"),
    "CsVecSum": (["vector", "int"], "vector"),
    "CsVecMean": (["vector", "int"], "vector"),
    "CsCorr": (["float", "float", "int"], "float"),
    "CsRangeMask": (["float", "const_float", "const_float", "float", "float", "int"], "float"),
    "CsNeutralize": (["float", "float", "int"], "vector"),
    "CsMultiNeutralize": (["vector", "float", "int"], "vector"),
    "CsBucket": (["float", "const_int", "int"], "int"),
    "CsVecRangeMask": (["vector", "const_float", "const_float", "float", "float", "int"], "vector"),
    "CsQuadReg": (["float", "float", "int"], "vector"),
    "CsHHI": (["float", "int"], "float"),
    "CsGini": (["float", "int"], "float"),
    "CsEntropy": (["float", "int"], "float"),

    # === 归一化运算 ===
    "TsZscoreNorm": (["factor", "const_int", "const_bool", "const_bool"], "float"),
    "TsMedianNorm": (["factor", "const_int", "const_bool"], "float"),
    "TsMadNorm": (["factor", "const_int", "const_bool"], "float"),
}

# ============================================
# 取值范围分组提示（用于RL奖励）
# ============================================

RANGE_HINTS = {
    # 时序运算
    "TsSum": {"ranges": [None, "small_int", None]},
    "TsMean": {"ranges": [None, "small_int", None]},
    "TsMedian": {"ranges": [None, "small_int", None]},
    "TsShift": {"ranges": [None, "medium_int", None]},
    "TsDiff": {"ranges": [None, "medium_int", None]},
    "TsRet": {"ranges": [None, "medium_int", None]},
    "TsEma": {"ranges": [None, "small_int", None]},
    "TsQuantile": {"ranges": [None, "pct_float", "small_int", None]},
    "TsWinsorize": {"ranges": [None, "small_float", "small_int", None]},
    "TsObv": {"ranges": [None, None, "small_int", None]},
    "TsRankRaw": {"ranges": [None, "small_int", None]},
    "TsFixedTimeWindow": {"ranges": [None, "large_int", "large_int", "large_int", None]},
    
    # 标量运算
    "Round": {"ranges": [None, "precision_int"]},
    "BRound": {"ranges": [None, "precision_int"]},
    "Zero": {"ranges": [None, "small_float"]},
    
    # 向量运算
    "VecNew": {"ranges": ["small_int"]},
    "VecRollingMeanVec": {"ranges": [None, "small_int"]},
    "VecPctFrontPoint": {"ranges": [None, "pct_float"]},
    "VecPctBackPoint": {"ranges": [None, "pct_float"]},
    "VecRangeMask": {"ranges": [None, "pct_float", "pct_float", None, None]},
    "VecRangeFilter": {"ranges": [None, "pct_float", "pct_float"]},
    "VecSequence": {"ranges": ["medium_int", "medium_int", "small_int"]},
    "VecExpDecaySeq": {"ranges": ["small_int", "small_int"]},
    "VecStepSubvec": {"ranges": [None, "medium_int", "medium_int", "small_int"]},
    "VecOneHot": {"ranges": ["index_vector", "small_int"]},
    "VecOneHotMask": {"ranges": ["index_vector", "small_int", None, None]},
    
    # 截面运算
    "CsBucket": {"ranges": [None, "small_int", "group_id_int"]},
    "CsWinsorize": {"ranges": [None, "small_float", None]},
    "CsRangeMask": {"ranges": [None, "pct_float", "pct_float", None, None, "group_id_int"]},
    
    # 历史数据
    "HistDailyField": {"ranges": [None, "medium_int", "small_int"]},
    "HistDailyFieldVec": {"ranges": [None, "small_int"]},
    "HistLastSliceFieldVec": {"ranges": [None, "small_int"]},
    "HistTdVec": {"ranges": [None, "medium_int"]},
    "HistCDSIVec": {"ranges": [None, "medium_int", "small_int", "medium_int"]},
    "HistSliceFieldTens": {"ranges": [None, "small_int"]},
    "HistTdTens": {"ranges": [None, "medium_int", "small_int"]},
    "HistTdMedianVec": {"ranges": [None, "medium_int", "small_int"]},
    "HistTdMedianTens": {"ranges": [None, "medium_int", "small_int", "small_int"]},
    
    # 归一化运算
    "TsZscoreNorm": {"ranges": [None, "small_int", None, None]},
    "TsMedianNorm": {"ranges": [None, "small_int", None]},
    "TsMadNorm": {"ranges": [None, "small_int", None]},
}

# ============================================
# 取值范围兼容性（组间兼容分数）
# ============================================

RANGE_COMPATIBILITY = {
    # ===== 整数范围组兼容性 =====
    # small_int与其他int组
    ("small_int", "medium_int"): 0.8,    # 小范围到中等范围：轻微惩罚
    ("small_int", "large_int"): 0.5,     # 小范围到大范围：较大惩罚
    ("small_int", "group_id_int"): 0.7,  # 小整数到分组ID：中等惩罚
    ("small_int", "precision_int"): 0.6, # 小整数到精度：较大惩罚

    # medium_int与其他int组
    ("medium_int", "small_int"): 0.8,    # 中等到小范围：轻微惩罚
    ("medium_int", "large_int"): 0.7,    # 中等到大范围：中等惩罚
    ("medium_int", "group_id_int"): 0.6, # 中等到分组ID：较大惩罚
    ("medium_int", "precision_int"): 0.5,# 中等到精度：较大惩罚

    # large_int与其他int组
    ("large_int", "small_int"): 0.4,     # 大到小：严重惩罚
    ("large_int", "medium_int"): 0.6,    # 大到中：较大惩罚
    ("large_int", "group_id_int"): 0.5,  # 大到分组：较大惩罚
    ("large_int", "precision_int"): 0.3, # 大到精度：严重惩罚

    # group_id_int与其他int组
    ("group_id_int", "small_int"): 0.7,  # 分组到小：中等惩罚
    ("group_id_int", "medium_int"): 0.8, # 分组到中：轻微惩罚
    ("group_id_int", "large_int"): 0.6,  # 分组到大：较大惩罚
    ("group_id_int", "precision_int"): 0.5, # 分组到精度：较大惩罚

    # precision_int与其他int组
    ("precision_int", "small_int"): 0.6, # 精度到小：较大惩罚
    ("precision_int", "medium_int"): 0.7, # 精度到中：中等惩罚
    ("precision_int", "large_int"): 0.4,  # 精度到大：严重惩罚
    ("precision_int", "group_id_int"): 0.6, # 精度到分组：较大惩罚

    # ===== 浮点数范围组兼容性 =====
    # pct_float与其他float组
    ("pct_float", "small_float"): 0.7,   # 百分比到小浮点：中等惩罚
    ("pct_float", "medium_float"): 0.8,  # 百分比到中浮点：轻微惩罚
    ("pct_float", "price_like_float"): 0.5, # 百分比到价格：较大惩罚

    # small_float与其他float组
    ("small_float", "pct_float"): 0.7,   # 小浮点到百分比：中等惩罚
    ("small_float", "medium_float"): 0.9, # 小浮点到中浮点：轻微惩罚
    ("small_float", "price_like_float"): 0.6, # 小浮点到价格：较大惩罚

    # medium_float与其他float组
    ("medium_float", "pct_float"): 0.8,  # 中浮点到百分比：轻微惩罚
    ("medium_float", "small_float"): 0.9, # 中浮点到小浮点：轻微惩罚
    ("medium_float", "price_like_float"): 0.7, # 中浮点到价格：中等惩罚

    # price_like_float与其他float组
    ("price_like_float", "pct_float"): 0.5, # 价格到百分比：较大惩罚
    ("price_like_float", "small_float"): 0.6, # 价格到小浮点：较大惩罚
    ("price_like_float", "medium_float"): 0.8, # 价格到中浮点：轻微惩罚
}

# ============================================
# 特殊常量取值提示
# ============================================

CONSTANT_RANGES = {
    "small_int": (1, 30),            # 短期窗口、计数 (1-30分钟)
    "medium_int": (0, 240),          # 分钟偏移、索引 (一天240分钟)
    "large_int": (1000, 200000000),  # 时间戳、大索引
    "group_id_int": (0, 20),         # 分组ID (较少的分组数量)
    "precision_int": (-3, 3),        # 精度参数 (缩小范围)

    "pct_float": (0, 100),           # 百分比 (0-100%)
    "small_float": (0, 5),           # 小比率、阈值 (0-5)
    "price_like_float": (10, 1000),  # 价格范围
}

# ============================================
# 类型验证和评分工具
# ============================================

def validate_operator_signatures(operator_signatures, type_hierarchy):
    """
    验证算子签名中使用的类型是否都在类型系统中定义
    """
    valid_types = set(type_hierarchy.keys())
    issues = []

    for op_name, (arg_types, return_type) in operator_signatures.items():
        # 检查参数类型
        for i, arg_type in enumerate(arg_types):
            if arg_type not in valid_types:
                issues.append(f"算子 {op_name} 第{i+1}个参数类型 '{arg_type}' 未定义")

        # 检查返回值类型
        if return_type not in valid_types and return_type != "same_as_input":
            issues.append(f"算子 {op_name} 返回值类型 '{return_type}' 未定义")

    return issues

# 验证算子签名
validation_issues = validate_operator_signatures(OPERATOR_SIGNATURES, TYPE_HIERARCHY)
if validation_issues:
    print("发现类型定义问题：")
    for issue in validation_issues:
        print(f"  - {issue}")
else:
    print("所有算子签名中的类型都已正确定义！")
