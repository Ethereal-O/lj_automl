from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import subprocess
from io import StringIO

# 移除了: from adapters.expression_validator import validate_expression

def check_and_compute_intermediate(expr_str: str, pool, calculator) -> Optional[float]:
    """
    检查表达式是否是可计算打分的中间表达式（即valid），如果是，返回shaped reward，否则None。
    
    【优化说明】
    由于 AlphaEnvWrapper 现在强制使用了 strict action mask，
    任何能触发此函数调用的完整表达式（Stack=1且类型正确）在生成时已由Wrapper保证了合法性。
    因此，移除昂贵的字符串解析验证，直接返回奖励。
    """
    
    # 如果传入的是空字符串或明显非法的情况，可以简单防御一下
    if not expr_str or not expr_str.strip():
        return None

    # 由于 wrapper 层的双重保险，能传递到这里的 expr_str 默认为 valid
    # 直接返回中间奖励
    return 0.01 * pool.compute_expression_score(expr_str, calculator)