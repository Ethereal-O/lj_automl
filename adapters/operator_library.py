"""
AlphaQCM 算子库
包含所有可用算子的基本信息：名称、参数类型、输出类型

只保留核心信息，移除复杂的类型检查逻辑
"""

# ============================================
# 算子签名字典
# 格式：算子名称: ([参数类型列表], 输出类型)
# ============================================

OPERATOR_SIGNATURES = {
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
    "IfElse": (["bool", "expr", "expr"], "expr"),
    "If": (["bool", "expr"], "expr"),
    "IsHist": ([], "bool"),
    "IsToday": ([], "bool"),

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

    # === 向量运算 ===
    "VecNew": (["int"], "vector"),
    "VecLen": (["vector"], "float"),
    "VecInplacePush": (["vector", "float"], "vector"),
    "VecInplacePushVec": (["vector", "vector"], "vector"),
    "VecInplaceSort": (["vector"], "vector"),
    "VecInplaceSortByVec": (["vector", "vector"], "vector"),
    "VecPush": (["vector", "float"], "vector"),
    "VecPushVec": (["vector", "vector"], "vector"),
    "VecSort": (["vector"], "vector"),
    "VecInplacePushSortKV": (["vector", "float", "float"], "vector"),
    "VecSortByVec": (["vector", "vector"], "vector"),
    "VecSortIndex": (["vector"], "vector"),
    "VecPctFrontPoint": (["vector", "float"], "float"),
    "VecPctBackPoint": (["vector", "float"], "float"),
    "VecPctFrontSubvec": (["vector", "float"], "vector"),
    "VecPctBackSubvec": (["vector", "float"], "vector"),
    "VecPctSubvec": (["vector", "float", "float"], "vector"),
    "VecMean": (["vector"], "float"),
    "VecSum": (["vector"], "float"),
    "VecProduct": (["vector"], "float"),
    "VecMidIdxValue": (["vector"], "any"),
    "VecMedian": (["vector"], "float"),
    "VecStd": (["vector"], "float"),
    "VecSkew": (["vector"], "float"),
    "VecKurt": (["vector"], "float"),
    "VecArgmin": (["vector"], "float"),
    "VecArgmax": (["vector"], "float"),
    "VecMultiArgmin": (["vector"], "vector"),
    "VecMultiArgmax": (["vector"], "vector"),
    "VecArgminZig": (["vector"], "float"),
    "VecArgmaxZig": (["vector"], "float"),
    "VecArgminmaxZig": (["vector"], "float"),
    "VecValue": (["vector", "float"], "any"),
    "VecMultiValue": (["vector", "vector"], "vector"),
    "VecBackSubvec": (["vector", "float"], "vector"),
    "VecMin": (["vector"], "float"),
    "VecMax": (["vector"], "float"),
    "VecMinZig": (["vector"], "float"),
    "VecMaxZig": (["vector"], "float"),
    "VecMinmaxZig": (["vector"], "float"),
    "VecCumsumVec": (["vector", "float"], "vector"),
    "VecPeakNum": (["vector"], "float"),
    "VecValleyNum": (["vector"], "float"),
    "VecPeakNumZig": (["vector"], "float"),
    "VecValleyNumZig": (["vector"], "float"),
    "VecWaveUpMagZigVec": (["vector"], "vector"),
    "VecWaveDownMagZigVec": (["vector"], "vector"),
    "VecWaveUpLenZigVec": (["vector"], "vector"),
    "VecWaveDownLenZigVec": (["vector"], "vector"),
    "VecInv": (["vector"], "vector"),
    "VecLog": (["vector"], "vector"),
    "VecAdd": (["vector", "vector"], "vector"),
    "VecSub": (["vector", "vector"], "vector"),
    "VecMul": (["vector", "vector"], "vector"),
    "VecDiv": (["vector", "vector"], "vector"),
    "VecBCAdd": (["vector", "float"], "vector"),
    "VecBCSub": (["vector", "float"], "vector"),
    "VecBCMul": (["vector", "float"], "vector"),
    "VecBCDiv": (["vector", "float"], "vector"),
    "VecRet": (["vector", "vector"], "vector"),
    "VecLogRet": (["vector", "vector"], "vector"),
    "VecRollingMeanVec": (["vector", "const_int"], "vector"),
    "VecSlope": (["vector"], "float"),
    "VecNormalize": (["vector"], "vector"),
    "VecShannonEntropy": (["vector"], "float"),
    "VecCrossEntropy": (["vector", "vector"], "float"),
    "VecKLDivergence": (["vector", "vector"], "float"),
    "VecCorr": (["vector", "vector"], "float"),
    "VecCov": (["vector", "vector"], "float"),
    "VecRank": (["vector"], "vector"),
    "VecLessIndex": (["vector", "float"], "vector"),
    "VecGreaterIndex": (["vector", "float"], "vector"),
    "VecOneHot": (["vector", "int"], "vector"),
    # VecRangeMask扩展：3*2*4=24个变体（与CsRangeMask完全相同）
    # 格式：VecRangeMask{Border}{Op}{Pct}
    # Border: L(lower边)/U(upper边)/D(双边)
    # Op: K(keep要极值)/R(remove去极值)
    # Pct: 01(1%)/05(5%)/10(10%)/25(25%)

    # Lower边要极值：lower=0, upper=Pct
    "VecRangeMaskLK01": (["vector", "float", "float"], "vector"),  # lower=0, upper=1
    "VecRangeMaskLK05": (["vector", "float", "float"], "vector"),  # lower=0, upper=5
    "VecRangeMaskLK10": (["vector", "float", "float"], "vector"),  # lower=0, upper=10
    "VecRangeMaskLK25": (["vector", "float", "float"], "vector"),  # lower=0, upper=25

    # Lower边去极值：lower=Pct, upper=100
    "VecRangeMaskLR01": (["vector", "float", "float"], "vector"),  # lower=1, upper=100
    "VecRangeMaskLR05": (["vector", "float", "float"], "vector"),  # lower=5, upper=100
    "VecRangeMaskLR10": (["vector", "float", "float"], "vector"),  # lower=10, upper=100
    "VecRangeMaskLR25": (["vector", "float", "float"], "vector"),  # lower=25, upper=100

    # Upper边要极值：lower=100-Pct, upper=100
    "VecRangeMaskUK01": (["vector", "float", "float"], "vector"),  # lower=99, upper=100
    "VecRangeMaskUK05": (["vector", "float", "float"], "vector"),  # lower=95, upper=100
    "VecRangeMaskUK10": (["vector", "float", "float"], "vector"),  # lower=90, upper=100
    "VecRangeMaskUK25": (["vector", "float", "float"], "vector"),  # lower=75, upper=100

    # Upper边去极值：lower=0, upper=100-Pct
    "VecRangeMaskUR01": (["vector", "float", "float"], "vector"),  # lower=0, upper=99
    "VecRangeMaskUR05": (["vector", "float", "float"], "vector"),  # lower=0, upper=95
    "VecRangeMaskUR10": (["vector", "float", "float"], "vector"),  # lower=0, upper=90
    "VecRangeMaskUR25": (["vector", "float", "float"], "vector"),  # lower=0, upper=75

    # 双边要极值：lower=Pct, upper=100-Pct
    "VecRangeMaskDK01": (["vector", "float", "float"], "vector"),  # lower=1, upper=99
    "VecRangeMaskDK05": (["vector", "float", "float"], "vector"),  # lower=5, upper=95
    "VecRangeMaskDK10": (["vector", "float", "float"], "vector"),  # lower=10, upper=90
    "VecRangeMaskDK25": (["vector", "float", "float"], "vector"),  # lower=25, upper=75

    # 双边去极值：lower=0, upper=Pct 和 lower=100-Pct, upper=100
    "VecRangeMaskDR01": (["vector", "float", "float"], "vector"),  # lower=0, upper=1 + lower=99, upper=100
    "VecRangeMaskDR05": (["vector", "float", "float"], "vector"),  # lower=0, upper=5 + lower=95, upper=100
    "VecRangeMaskDR10": (["vector", "float", "float"], "vector"),  # lower=0, upper=10 + lower=90, upper=100
    "VecRangeMaskDR25": (["vector", "float", "float"], "vector"),  # lower=0, upper=25 + lower=75, upper=100

    # VecRangeFilter扩展：3*2*4=24个变体（与CsRangeMask完全相同）
    # 格式：VecRangeFilter{Border}{Op}{Pct}
    # Border: L(lower边)/U(upper边)/D(双边)
    # Op: K(keep要极值)/R(remove去极值)
    # Pct: 01(1%)/05(5%)/10(10%)/25(25%)

    # Lower边要极值：lower=0, upper=Pct
    "VecRangeFilterLK01": (["vector", "float", "float"], "vector"),  # lower=0, upper=1
    "VecRangeFilterLK05": (["vector", "float", "float"], "vector"),  # lower=0, upper=5
    "VecRangeFilterLK10": (["vector", "float", "float"], "vector"),  # lower=0, upper=10
    "VecRangeFilterLK25": (["vector", "float", "float"], "vector"),  # lower=0, upper=25

    #：lower=Pct, upper=100
    "VecRangeFilterLR01": (["vector", "float", "float"], "vector"),  # lower=1, upper=100
    "VecRangeFilterLR05": (["vector", "float", "float"], "vector"),  # lower=5, upper=100
    "VecRangeFilterLR10": (["vector", "float", "float"], "vector"),  # lower=10, upper=100
    "VecRangeFilterLR25": (["vector", "float", "float"], "vector"),  # lower=25, upper=100

    # Upper边要极值：lower=100-Pct, upper=100
    "VecRangeFilterUK01": (["vector", "float", "float"], "vector"),  # lower=99, upper=100
    "VecRangeFilterUK05": (["vector", "float", "float"], "vector"),  # lower=95, upper=100
    "VecRangeFilterUK10": (["vector", "float", "float"], "vector"),  # lower=90, upper=100
    "VecRangeFilterUK25": (["vector", "float", "float"], "vector"),  # lower=75, upper=100

    # Upper边去极值：lower=0, upper=100-Pct
    "VecRangeFilterUR01": (["vector", "float", "float"], "vector"),  # lower=0, upper=99
    "VecRangeFilterUR05": (["vector", "float", "float"], "vector"),  # lower=0, upper=95
    "VecRangeFilterUR10": (["vector", "float", "float"], "vector"),  # lower=0, upper=90
    "VecRangeFilterUR25": (["vector", "float", "float"], "vector"),  # lower=0, upper=75

    # 双边要极值：lower=Pct, upper=100-Pct
    "VecRangeFilterDK01": (["vector", "float", "float"], "vector"),  # lower=1, upper=99
    "VecRangeFilterDK05": (["vector", "float", "float"], "vector"),  # lower=5, upper=95
    "VecRangeFilterDK10": (["vector", "float", "float"], "vector"),  # lower=10, upper=90
    "VecRangeFilterDK25": (["vector", "float", "float"], "vector"),  # lower=25, upper=75

    # 双边去极值：lower=0, upper=Pct 和 lower=100-Pct, upper=100
    "VecRangeFilterDR01": (["vector", "float", "float"], "vector"),  # lower=0, upper=1 + lower=99, upper=100
    "VecRangeFilterDR05": (["vector", "float", "float"], "vector"),  # lower=0, upper=5 + lower=95, upper=100
    "VecRangeFilterDR10": (["vector", "float", "float"], "vector"),  # lower=0, upper=10 + lower=90, upper=100
    "VecRangeFilterDR25": (["vector", "float", "float"], "vector"),  # lower=0, upper=25 + lower=75, upper=100
    "VecMaskFilter": (["vector", "vector", "bool"], "vector"),
    "VecSpreadMask": (["vector", "int", "int", "int"], "vector"),
    "VecQuadReg": (["vector", "vector"], "vector"),
    "VecZscore": (["vector"], "vector"),
    "VecOneHotMask": (["vector", "int", "float", "float"], "vector"),
    "VecSequence": (["int", "int", "int"], "vector"),
    "VecExpDecaySeq": (["int", "int"], "vector"),
    "VecReverse": (["vector"], "vector"),
    "VecNegative": (["vector"], "vector"),
    "VecStepSubvec": (["vector", "int", "int", "int"], "vector"),
    "VecValueIdx": (["vector", "float"], "vector"),
    "VecFwdFill": (["vector", "float"], "vector"),
    "VecBwdFill": (["vector", "float"], "vector"),
    "VecFill": (["vector", "float", "float"], "vector"),
    "VecFillByVec": (["vector", "float", "vector"], "vector"),
    "VecMask": (["vector", "vector", "float", "bool"], "vector"),
    "VecMaskByVec": (["vector", "vector", "vector", "bool"], "vector"),
    "VecRankRaw": (["vector", "bool"], "vector"),
    "VecDropValues": (["vector", "vector"], "vector"),
    "VecKeepValues": (["vector", "vector"], "vector"),
    "VecInValues": (["vector", "vector", "bool"], "vector"),
    "VecBool": (["vector"], "vector"),
    "VecNot": (["vector"], "vector"),
    "VecBCEqual": (["vector", "float", "bool"], "vector"),
    "VecBCGreater": (["vector", "float"], "vector"),
    "VecBCLess": (["vector", "float"], "vector"),
    "VecBCNotGreater": (["vector", "float"], "vector"),
    "VecBCNotLess": (["vector", "float"], "vector"),
    "VecEqual": (["vector", "vector", "bool"], "vector"),
    "VecGreater": (["vector", "vector"], "vector"),
    "VecLess": (["vector", "vector"], "vector"),
    "VecNotGreater": (["vector", "vector"], "vector"),
    "VecNotLess": (["vector", "vector"], "vector"),
    "VecAny": (["vector"], "bool"),
    "VecAll": (["vector"], "bool"),
    "VecOr": (["vector", "vector"], "vector"),
    "VecXor": (["vector", "vector"], "vector"),
    "VecAnd": (["vector", "vector"], "vector"),
    "VecPeak": (["vector", "float", "float", "float"], "vector"),
    "VecPeakIdx": (["vector", "float", "float", "float"], "vector"),
    "VecPeakCnt": (["vector", "float", "float", "float"], "int"),
    "VecValley": (["vector", "float", "float", "float"], "vector"),
    "VecValleyIdx": (["vector", "float", "float", "float"], "vector"),
    "VecValleyCnt": (["vector", "float", "float", "float"], "int"),

    # 单参数时序算子：false不跨天（6个窗口）
    "TsSum1F": (["float"], "float"), "TsSum5F": (["float"], "float"),
    "TsSum20F": (["float"], "float"), "TsSum60F": (["float"], "float"),
    "TsSum120F": (["float"], "float"), "TsSum240F": (["float"], "float"),

    "TsMean1F": (["float"], "float"), "TsMean5F": (["float"], "float"),
    "TsMean20F": (["float"], "float"), "TsMean60F": (["float"], "float"),
    "TsMean120F": (["float"], "float"), "TsMean240F": (["float"], "float"),

    "TsMedian1F": (["float"], "float"), "TsMedian5F": (["float"], "float"),
    "TsMedian20F": (["float"], "float"), "TsMedian60F": (["float"], "float"),
    "TsMedian120F": (["float"], "float"), "TsMedian240F": (["float"], "float"),

    "TsArgmax1F": (["float"], "float"), "TsArgmax5F": (["float"], "float"),
    "TsArgmax20F": (["float"], "float"), "TsArgmax60F": (["float"], "float"),
    "TsArgmax120F": (["float"], "float"), "TsArgmax240F": (["float"], "float"),

    "TsArgmin1F": (["float"], "float"), "TsArgmin5F": (["float"], "float"),
    "TsArgmin20F": (["float"], "float"), "TsArgmin60F": (["float"], "float"),
    "TsArgmin120F": (["float"], "float"), "TsArgmin240F": (["float"], "float"),

    "TsMax1F": (["float"], "float"), "TsMax5F": (["float"], "float"),
    "TsMax20F": (["float"], "float"), "TsMax60F": (["float"], "float"),
    "TsMax120F": (["float"], "float"), "TsMax240F": (["float"], "float"),

    "TsMin1F": (["float"], "float"), "TsMin5F": (["float"], "float"),
    "TsMin20F": (["float"], "float"), "TsMin60F": (["float"], "float"),
    "TsMin120F": (["float"], "float"), "TsMin240F": (["float"], "float"),

    "TsShift1F": (["float"], "float"), "TsShift5F": (["float"], "float"),
    "TsShift20F": (["float"], "float"), "TsShift60F": (["float"], "float"),
    "TsShift120F": (["float"], "float"), "TsShift240F": (["float"], "float"),

    "TsDiff1F": (["float"], "float"), "TsDiff5F": (["float"], "float"),
    "TsDiff20F": (["float"], "float"), "TsDiff60F": (["float"], "float"),
    "TsDiff120F": (["float"], "float"), "TsDiff240F": (["float"], "float"),

    "TsRet1F": (["float"], "float"), "TsRet5F": (["float"], "float"),
    "TsRet20F": (["float"], "float"), "TsRet60F": (["float"], "float"),
    "TsRet120F": (["float"], "float"), "TsRet240F": (["float"], "float"),

    "TsLogRet1F": (["float"], "float"), "TsLogRet5F": (["float"], "float"),
    "TsLogRet20F": (["float"], "float"), "TsLogRet60F": (["float"], "float"),
    "TsLogRet120F": (["float"], "float"), "TsLogRet240F": (["float"], "float"),

    "TsAbsRet1F": (["float"], "float"), "TsAbsRet5F": (["float"], "float"),
    "TsAbsRet20F": (["float"], "float"), "TsAbsRet60F": (["float"], "float"),
    "TsAbsRet120F": (["float"], "float"), "TsAbsRet240F": (["float"], "float"),

    "TsVar1F": (["float"], "float"), "TsVar5F": (["float"], "float"),
    "TsVar20F": (["float"], "float"), "TsVar60F": (["float"], "float"),
    "TsVar120F": (["float"], "float"), "TsVar240F": (["float"], "float"),

    "TsStd1F": (["float"], "float"), "TsStd5F": (["float"], "float"),
    "TsStd20F": (["float"], "float"), "TsStd60F": (["float"], "float"),
    "TsStd120F": (["float"], "float"), "TsStd240F": (["float"], "float"),

    "TsSkew1F": (["float"], "float"), "TsSkew5F": (["float"], "float"),
    "TsSkew20F": (["float"], "float"), "TsSkew60F": (["float"], "float"),
    "TsSkew120F": (["float"], "float"), "TsSkew240F": (["float"], "float"),

    "TsKurt1F": (["float"], "float"), "TsKurt5F": (["float"], "float"),
    "TsKurt20F": (["float"], "float"), "TsKurt60F": (["float"], "float"),
    "TsKurt120F": (["float"], "float"), "TsKurt240F": (["float"], "float"),

    "TsEma1F": (["float"], "float"), "TsEma5F": (["float"], "float"),
    "TsEma20F": (["float"], "float"), "TsEma60F": (["float"], "float"),
    "TsEma120F": (["float"], "float"), "TsEma240F": (["float"], "float"),

    "TsZscore1F": (["float"], "float"), "TsZscore5F": (["float"], "float"),
    "TsZscore20F": (["float"], "float"), "TsZscore60F": (["float"], "float"),
    "TsZscore120F": (["float"], "float"), "TsZscore240F": (["float"], "float"),

    "TsSlope1F": (["float"], "float"), "TsSlope5F": (["float"], "float"),
    "TsSlope20F": (["float"], "float"), "TsSlope60F": (["float"], "float"),
    "TsSlope120F": (["float"], "float"), "TsSlope240F": (["float"], "float"),

    "TsRank1F": (["float"], "float"), "TsRank5F": (["float"], "float"),
    "TsRank20F": (["float"], "float"), "TsRank60F": (["float"], "float"),
    "TsRank120F": (["float"], "float"), "TsRank240F": (["float"], "float"),

    "TsMad1F": (["float"], "float"), "TsMad5F": (["float"], "float"),
    "TsMad20F": (["float"], "float"), "TsMad60F": (["float"], "float"),
    "TsMad120F": (["float"], "float"), "TsMad240F": (["float"], "float"),

    "TsWeightCenter1F": (["float"], "float"), "TsWeightCenter5F": (["float"], "float"),
    "TsWeightCenter20F": (["float"], "float"), "TsWeightCenter60F": (["float"], "float"),
    "TsWeightCenter120F": (["float"], "float"), "TsWeightCenter240F": (["float"], "float"),

    "TsHHI1F": (["float"], "float"), "TsHHI5F": (["float"], "float"),
    "TsHHI20F": (["float"], "float"), "TsHHI60F": (["float"], "float"),
    "TsHHI120F": (["float"], "float"), "TsHHI240F": (["float"], "float"),

    "TsGini1F": (["float"], "float"), "TsGini5F": (["float"], "float"),
    "TsGini20F": (["float"], "float"), "TsGini60F": (["float"], "float"),
    "TsGini120F": (["float"], "float"), "TsGini240F": (["float"], "float"),

    "TsEntropy1F": (["float"], "float"), "TsEntropy5F": (["float"], "float"),
    "TsEntropy20F": (["float"], "float"), "TsEntropy60F": (["float"], "float"),
    "TsEntropy120F": (["float"], "float"), "TsEntropy240F": (["float"], "float"),

    # 时序算子：true跨天（4个窗口）
    "TsSum1T": (["float"], "float"), "TsSum2T": (["float"], "float"),
    "TsSum5T": (["float"], "float"), "TsSum20T": (["float"], "float"),

    "TsMean1T": (["float"], "float"), "TsMean2T": (["float"], "float"),
    "TsMean5T": (["float"], "float"), "TsMean20T": (["float"], "float"),

    "TsMedian1T": (["float"], "float"), "TsMedian2T": (["float"], "float"),
    "TsMedian5T": (["float"], "float"), "TsMedian20T": (["float"], "float"),

    "TsArgmax1T": (["float"], "float"), "TsArgmax2T": (["float"], "float"),
    "TsArgmax5T": (["float"], "float"), "TsArgmax20T": (["float"], "float"),

    "TsArgmin1T": (["float"], "float"), "TsArgmin2T": (["float"], "float"),
    "TsArgmin5T": (["float"], "float"), "TsArgmin20T": (["float"], "float"),

    "TsMax1T": (["float"], "float"), "TsMax2T": (["float"], "float"),
    "TsMax5T": (["float"], "float"), "TsMax20T": (["float"], "float"),

    "TsMin1T": (["float"], "float"), "TsMin2T": (["float"], "float"),
    "TsMin5T": (["float"], "float"), "TsMin20T": (["float"], "float"),

    "TsShift1T": (["float"], "float"), "TsShift2T": (["float"], "float"),
    "TsShift5T": (["float"], "float"), "TsShift20T": (["float"], "float"),

    "TsDiff1T": (["float"], "float"), "TsDiff2T": (["float"], "float"),
    "TsDiff5T": (["float"], "float"), "TsDiff20T": (["float"], "float"),

    "TsRet1T": (["float"], "float"), "TsRet2T": (["float"], "float"),
    "TsRet5T": (["float"], "float"), "TsRet20T": (["float"], "float"),

    "TsLogRet1T": (["float"], "float"), "TsLogRet2T": (["float"], "float"),
    "TsLogRet5T": (["float"], "float"), "TsLogRet20T": (["float"], "float"),

    "TsAbsRet1T": (["float"], "float"), "TsAbsRet2T": (["float"], "float"),
    "TsAbsRet5T": (["float"], "float"), "TsAbsRet20T": (["float"], "float"),

    "TsVar1T": (["float"], "float"), "TsVar2T": (["float"], "float"),
    "TsVar5T": (["float"], "float"), "TsVar20T": (["float"], "float"),

    "TsStd1T": (["float"], "float"), "TsStd2T": (["float"], "float"),
    "TsStd5T": (["float"], "float"), "TsStd20T": (["float"], "float"),

    "TsSkew1T": (["float"], "float"), "TsSkew2T": (["float"], "float"),
    "TsSkew5T": (["float"], "float"), "TsSkew20T": (["float"], "float"),

    "TsKurt1T": (["float"], "float"), "TsKurt2T": (["float"], "float"),
    "TsKurt5T": (["float"], "float"), "TsKurt20T": (["float"], "float"),

    "TsEma1T": (["float"], "float"), "TsEma2T": (["float"], "float"),
    "TsEma5T": (["float"], "float"), "TsEma20T": (["float"], "float"),

    "TsZscore1T": (["float"], "float"), "TsZscore2T": (["float"], "float"),
    "TsZscore5T": (["float"], "float"), "TsZscore20T": (["float"], "float"),

    "TsSlope1T": (["float"], "float"), "TsSlope2T": (["float"], "float"),
    "TsSlope5T": (["float"], "float"), "TsSlope20T": (["float"], "float"),

    "TsRank1T": (["float"], "float"), "TsRank2T": (["float"], "float"),
    "TsRank5T": (["float"], "float"), "TsRank20T": (["float"], "float"),

    "TsMad1T": (["float"], "float"), "TsMad2T": (["float"], "float"),
    "TsMad5T": (["float"], "float"), "TsMad20T": (["float"], "float"),

    "TsWeightCenter1T": (["float"], "float"), "TsWeightCenter2T": (["float"], "float"),
    "TsWeightCenter5T": (["float"], "float"), "TsWeightCenter20T": (["float"], "float"),

    "TsHHI1T": (["float"], "float"), "TsHHI2T": (["float"], "float"),
    "TsHHI5T": (["float"], "float"), "TsHHI20T": (["float"], "float"),

    "TsGini1T": (["float"], "float"), "TsGini2T": (["float"], "float"),
    "TsGini5T": (["float"], "float"), "TsGini20T": (["float"], "float"),

    "TsEntropy1T": (["float"], "float"), "TsEntropy2T": (["float"], "float"),
    "TsEntropy5T": (["float"], "float"), "TsEntropy20T": (["float"], "float"),

    # 双参数时序算子：false不跨天（6个窗口）
    "TsCorr1F": (["float", "float"], "float"), "TsCorr5F": (["float", "float"], "float"),
    "TsCorr20F": (["float", "float"], "float"), "TsCorr60F": (["float", "float"], "float"),
    "TsCorr120F": (["float", "float"], "float"), "TsCorr240F": (["float", "float"], "float"),

    "TsCov1F": (["float", "float"], "float"), "TsCov5F": (["float", "float"], "float"),
    "TsCov20F": (["float", "float"], "float"), "TsCov60F": (["float", "float"], "float"),
    "TsCov120F": (["float", "float"], "float"), "TsCov240F": (["float", "float"], "float"),

    "TsObv1F": (["float", "float"], "float"), "TsObv5F": (["float", "float"], "float"),
    "TsObv20F": (["float", "float"], "float"), "TsObv60F": (["float", "float"], "float"),
    "TsObv120F": (["float", "float"], "float"), "TsObv240F": (["float", "float"], "float"),

    # 双参数时序算子：true跨天（4个窗口）
    "TsCorr1T": (["float", "float"], "float"), "TsCorr2T": (["float", "float"], "float"),
    "TsCorr5T": (["float", "float"], "float"), "TsCorr20T": (["float", "float"], "float"),

    "TsCov1T": (["float", "float"], "float"), "TsCov2T": (["float", "float"], "float"),
    "TsCov5T": (["float", "float"], "float"), "TsCov20T": (["float", "float"], "float"),

    "TsObv1T": (["float", "float"], "float"), "TsObv2T": (["float", "float"], "float"),
    "TsObv5T": (["float", "float"], "float"), "TsObv20T": (["float", "float"], "float"),

    # 其他复杂时序算子：参数写死版本

    # 三参数算子：quantile百分位（原来5参数，现在3参数）
    "TsQuantile25F": (["float", "float"], "float"), "TsQuantile50F": (["float", "float"], "float"),
    "TsQuantile75F": (["float", "float"], "float"), "TsQuantile90F": (["float", "float"], "float"),

    "TsQuantile25T": (["float", "float"], "float"), "TsQuantile50T": (["float", "float"], "float"),
    "TsQuantile75T": (["float", "float"], "float"), "TsQuantile90T": (["float", "float"], "float"),

    # 三参数算子：winsorize百分位（原来5参数，现在3参数）
    "TsWinsorize5F": (["float", "float"], "float"), "TsWinsorize10F": (["float", "float"], "float"),
    "TsWinsorize25F": (["float", "float"], "float"), "TsWinsorize5T": (["float", "float"], "float"),
    "TsWinsorize10T": (["float", "float"], "float"), "TsWinsorize25T": (["float", "float"], "float"),

    # 三参数算子：fill（原来5参数，现在3参数）
    "TsFill1F": (["float", "float"], "float"), "TsFill5F": (["float", "float"], "float"),
    "TsFill20F": (["float", "float"], "float"), "TsFill1T": (["float", "float"], "float"),
    "TsFill5T": (["float", "float"], "float"), "TsFill20T": (["float", "float"], "float"),

    # 三参数算子：weight mean（原来5参数，现在3参数）
    "TsWeightMean1F": (["float", "float"], "float"), "TsWeightMean5F": (["float", "float"], "float"),
    "TsWeightMean20F": (["float", "float"], "float"), "TsWeightMean1T": (["float", "float"], "float"),
    "TsWeightMean5T": (["float", "float"], "float"), "TsWeightMean20T": (["float", "float"], "float"),

    # 四参数算子：quad reg（原来6参数，现在4参数）
    "TsQuadReg1F": (["float", "float", "float"], "vector"), "TsQuadReg5F": (["float", "float", "float"], "vector"),
    "TsQuadReg20F": (["float", "float", "float"], "vector"), "TsQuadReg1T": (["float", "float", "float"], "vector"),
    "TsQuadReg5T": (["float", "float", "float"], "vector"), "TsQuadReg20T": (["float", "float", "float"], "vector"),

    # 二参数算子：FFT（原来4参数，现在2参数）
    "TsFFT1F": (["float"], "vector"), "TsFFT5F": (["float"], "vector"),
    "TsFFT20F": (["float"], "vector"), "TsFFT1T": (["float"], "vector"),
    "TsFFT5T": (["float"], "vector"), "TsFFT20T": (["float"], "vector"),

    # 四参数算子：ATR（原来7参数，现在4参数）
    "TsATR1F": (["float", "float", "float"], "float"), "TsATR5F": (["float", "float", "float"], "float"),
    "TsATR20F": (["float", "float", "float"], "float"), "TsATR1T": (["float", "float", "float"], "float"),
    "TsATR5T": (["float", "float", "float"], "float"), "TsATR20T": (["float", "float", "float"], "float"),

    # 二参数算子：rank raw（原来4参数，现在2参数）
    "TsRankRaw1F": (["float"], "int"), "TsRankRaw5F": (["float"], "int"),
    "TsRankRaw20F": (["float"], "int"), "TsRankRaw1T": (["float"], "int"),
    "TsRankRaw5T": (["float"], "int"), "TsRankRaw20T": (["float"], "int"),


    # === 截面运算 ===
    "CsVec": (["float", "int"], "vector"),
    "CsSum": (["float", "int"], "float"),
    "CsMean": (["float", "int"], "float"),
    "CsMedian": (["float", "int"], "float"),
    "CsVar": (["float", "int"], "float"),
    "CsStd": (["float", "int"], "float"),
    "CsRank": (["float", "int"], "float"),
    "CsZscore": (["float", "int"], "float"),
    # CsWinsorize扩展：std_ratio选0.5的倍数
    "CsWinsorize05": (["float", "int"], "float"),  # std_ratio=0.5
    "CsWinsorize10": (["float", "int"], "float"),  # std_ratio=1.0
    "CsWinsorize15": (["float", "int"], "float"),  # std_ratio=1.5
    "CsWinsorize20": (["float", "int"], "float"),  # std_ratio=2.0
    "CsWinsorize25": (["float", "int"], "float"),  # std_ratio=2.5
    "CsWinsorize30": (["float", "int"], "float"),  # std_ratio=3.0
    "CsVecSum": (["vector", "int"], "vector"),
    "CsVecMean": (["vector", "int"], "vector"),
    "CsCorr": (["float", "float", "int"], "float"),
    # CsRangeMask扩展：3*2*4=24个变体
    # 格式：CsRangeMask{Border}{Op}{Pct}，参数减少为4个（原来6个）
    # Border: L(lower边)/U(upper边)/D(双边)
    # Op: K(keep要极值)/R(remove去极值)
    # Pct: 01(1%)/05(5%)/10(10%)/25(25%)

    # Lower边要极值：lower=0, upper=Pct
    "CsRangeMaskLK01": (["float", "float", "float", "int"], "float"),  # lower=0, upper=1
    "CsRangeMaskLK05": (["float", "float", "float", "int"], "float"),  # lower=0, upper=5
    "CsRangeMaskLK10": (["float", "float", "float", "int"], "float"),  # lower=0, upper=10
    "CsRangeMaskLK25": (["float", "float", "float", "int"], "float"),  # lower=0, upper=25

    # Lower边去极值：lower=Pct, upper=100
    "CsRangeMaskLR01": (["float", "float", "float", "int"], "float"),  # lower=1, upper=100
    "CsRangeMaskLR05": (["float", "float", "float", "int"], "float"),  # lower=5, upper=100
    "CsRangeMaskLR10": (["float", "float", "float", "int"], "float"),  # lower=10, upper=100
    "CsRangeMaskLR25": (["float", "float", "float", "int"], "float"),  # lower=25, upper=100

    # Upper边要极值：lower=100-Pct, upper=100
    "CsRangeMaskUK01": (["float", "float", "float", "int"], "float"),  # lower=99, upper=100
    "CsRangeMaskUK05": (["float", "float", "float", "int"], "float"),  # lower=95, upper=100
    "CsRangeMaskUK10": (["float", "float", "float", "int"], "float"),  # lower=90, upper=100
    "CsRangeMaskUK25": (["float", "float", "float", "int"], "float"),  # lower=75, upper=100

    # Upper边去极值：lower=0, upper=100-Pct
    "CsRangeMaskUR01": (["float", "float", "float", "int"], "float"),  # lower=0, upper=99
    "CsRangeMaskUR05": (["float", "float", "float", "int"], "float"),  # lower=0, upper=95
    "CsRangeMaskUR10": (["float", "float", "float", "int"], "float"),  # lower=0, upper=90
    "CsRangeMaskUR25": (["float", "float", "float", "int"], "float"),  # lower=0, upper=75

    # 双边要极值：lower=Pct, upper=100-Pct
    "CsRangeMaskDK01": (["float", "float", "float", "int"], "float"),  # lower=1, upper=99
    "CsRangeMaskDK05": (["float", "float", "float", "int"], "float"),  # lower=5, upper=95
    "CsRangeMaskDK10": (["float", "float", "float", "int"], "float"),  # lower=10, upper=90
    "CsRangeMaskDK25": (["float", "float", "float", "int"], "float"),  # lower=25, upper=75

    # 双边去极值：lower=0, upper=Pct 和 lower=100-Pct, upper=100
    "CsRangeMaskDR01": (["float", "float", "float", "int"], "float"),  # lower=0, upper=1 + lower=99, upper=100
    "CsRangeMaskDR05": (["float", "float", "float", "int"], "float"),  # lower=0, upper=5 + lower=95, upper=100
    "CsRangeMaskDR10": (["float", "float", "float", "int"], "float"),  # lower=0, upper=10 + lower=90, upper=100
    "CsRangeMaskDR25": (["float", "float", "float", "int"], "float"),  # lower=0, upper=25 + lower=75, upper=100
    "CsNeutralize": (["float", "float", "int"], "vector"),
    "CsMultiNeutralize": (["vector", "float", "int"], "vector"),
    "CsBucket": (["float", "const_int", "int"], "int"),
    # CsVecRangeMask扩展：3*2*4=24个变体（与CsRangeMask完全相同）
    # 格式：CsVecRangeMask{Border}{Op}{Pct}
    # Border: L(lower边)/U(upper边)/D(双边)
    # Op: K(keep要极值)/R(remove去极值)
    # Pct: 01(1%)/05(5%)/10(10%)/25(25%)

    # Lower边要极值：lower=0, upper=Pct
    "CsVecRangeMaskLK01": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=1
    "CsVecRangeMaskLK05": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=5
    "CsVecRangeMaskLK10": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=10
    "CsVecRangeMaskLK25": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=25

    # Lower边去极值：lower=Pct, upper=100
    "CsVecRangeMaskLR01": (["vector", "float", "float", "int"], "vector"),  # lower=1, upper=100
    "CsVecRangeMaskLR05": (["vector", "float", "float", "int"], "vector"),  # lower=5, upper=100
    "CsVecRangeMaskLR10": (["vector", "float", "float", "int"], "vector"),  # lower=10, upper=100
    "CsVecRangeMaskLR25": (["vector", "float", "float", "int"], "vector"),  # lower=25, upper=100

    # Upper边要极值：lower=100-Pct, upper=100
    "CsVecRangeMaskUK01": (["vector", "float", "float", "int"], "vector"),  # lower=99, upper=100
    "CsVecRangeMaskUK05": (["vector", "float", "float", "int"], "vector"),  # lower=95, upper=100
    "CsVecRangeMaskUK10": (["vector", "float", "float", "int"], "vector"),  # lower=90, upper=100
    "CsVecRangeMaskUK25": (["vector", "float", "float", "int"], "vector"),  # lower=75, upper=100

    # Upper边去极值：lower=0, upper=100-Pct
    "CsVecRangeMaskUR01": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=99
    "CsVecRangeMaskUR05": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=95
    "CsVecRangeMaskUR10": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=90
    "CsVecRangeMaskUR25": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=75

    # 双边要极值：lower=Pct, upper=100-Pct
    "CsVecRangeMaskDK01": (["vector", "float", "float", "int"], "vector"),  # lower=1, upper=99
    "CsVecRangeMaskDK05": (["vector", "float", "float", "int"], "vector"),  # lower=5, upper=95
    "CsVecRangeMaskDK10": (["vector", "float", "float", "int"], "vector"),  # lower=10, upper=90
    "CsVecRangeMaskDK25": (["vector", "float", "float", "int"], "vector"),  # lower=25, upper=75

    # 双边去极值：lower=0, upper=Pct 和 lower=100-Pct, upper=100
    "CsVecRangeMaskDR01": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=1 + lower=99, upper=100
    "CsVecRangeMaskDR05": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=5 + lower=95, upper=100
    "CsVecRangeMaskDR10": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=10 + lower=90, upper=100
    "CsVecRangeMaskDR25": (["vector", "float", "float", "int"], "vector"),  # lower=0, upper=25 + lower=75, upper=100
    "CsQuadReg": (["float", "float", "int"], "vector"),
    "CsHHI": (["float", "int"], "float"),
    "CsGini": (["float", "int"], "float"),
    "CsEntropy": (["float", "int"], "float"),

}



if __name__ == "__main__":
    # 简单测试
    print(f"总共加载了 {len(OPERATOR_SIGNATURES)} 个算子")

