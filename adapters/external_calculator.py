from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import subprocess
from io import StringIO


def check_and_compute_intermediate(expr_str: str, pool, calculator) -> Optional[float]:
    """
    检查表达式是否是可计算打分的中间表达式（即valid），如果是，返回shaped reward，否则None。
    中间奖励用于episode结束后的总奖励，不用于即时反馈。
    """
    # 使用expression_validator检查是否valid
    from adapters.expression_validator import validate_expression
    valid, msg = validate_expression(expr_str)
    if not valid:
        return None

    # 返回shaped reward，如鼓励继续的正值
    return 0.01
