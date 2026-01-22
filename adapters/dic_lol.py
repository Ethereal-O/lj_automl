result_dict = {
    # ============ Preload 字段 ============
    "Preload.Concepts": ("vector", []),
    "Preload.ConceptSize": ("int", []),
    "Preload.Universe": ("int", []),
    "Preload.UniverseSize": ("int", []),
    "Preload.SW1": ("int", []),
    "Preload.SW1Size": ("int", []),
    "Preload.SW2": ("int", []),
    "Preload.SW2Size": ("int", []),
    "Preload.SW3": ("int", []),
    "Preload.SW3Size": ("int", []),
    "Preload.Category": ("int", []),
    "Preload.Exchange": ("int", []),
    "Preload.Board": ("int", []),
    "Preload.IsSH": ("bool", []),
    "Preload.IsSZ": ("bool", []),
    "Preload.IsSHMain": ("bool", []),
    "Preload.IsSHStar": ("bool", []),
    "Preload.IsSZMain": ("bool", []),
    "Preload.IsSZGem": ("bool", []),
    "Preload.Treatment": ("int", []),
    "Preload.IsNormal": ("bool", []),
    "Preload.FreeMarketCap": ("float", []),
    "Preload.BarraSize": ("float", []),
    "Preload.BarraIndustry": ("int", []),
    "Preload.BarraIndustrySize": ("int", []),
    "Preload.BETA": ("float", []),
    "Preload.MOMENTUM": ("float", []),
    "Preload.SIZE": ("float", []),
    "Preload.EARNYILD": ("float", []),
    "Preload.RESVOL": ("float", []),
    "Preload.GROWTH": ("float", []),
    "Preload.BTOP": ("float", []),
    "Preload.LEVERAGE": ("float", []),
    "Preload.LIQUIDTY": ("float", []),
    "Preload.MIDCAP": ("float", []),
    "Preload.DIVYILD": ("float", []),
    "Preload.EARNQLTY": ("float", []),
    "Preload.EARNVAR": ("float", []),
    "Preload.INVSQLTY": ("float", []),
    "Preload.LTREVRSL": ("float", []),
    "Preload.PROFIT": ("float", []),
    "Preload.ANALSENTI": ("float", []),
    "Preload.INDMOM": ("float", []),
    "Preload.SEASON": ("float", []),
    "Preload.STREVRSL": ("float", []),
    "Preload.AGRICULTURE": ("float", []),
    "Preload.AUTOMOBILES": ("float", []),
    "Preload.BANKS": ("float", []),
    "Preload.BUILDMATER": ("float", []),
    "Preload.CHEMICALS": ("float", []),
    "Preload.COMMERCE": ("float", []),
    "Preload.COMPUTERS": ("float", []),
    "Preload.CONGLOMERATES": ("float", []),
    "Preload.CONSTRDECOR": ("float", []),
    "Preload.DEFENSE": ("float", []),
    "Preload.ELECTRICALEQUIP": ("float", []),
    "Preload.ELECTRONICS": ("float", []),
    "Preload.FOODBEVERAGES": ("float", []),
    "Preload.HEALTHCARE": ("float", []),
    "Preload.HOMEAPPLIANCES": ("float", []),
    "Preload.LEISURE": ("float", []),
    "Preload.LIGHTINDUSTRY": ("float", []),
    "Preload.MACHINEEQUIP": ("float", []),
    "Preload.MEDIA": ("float", []),
    "Preload.MINING": ("float", []),
    "Preload.NONBANKFINAN": ("float", []),
    "Preload.NONFERROUSMETALS": ("float", []),
    "Preload.REALESTATE": ("float", []),
    "Preload.STEEL": ("float", []),
    "Preload.TELECOMS": ("float", []),
    "Preload.TEXTILEGARMENT": ("float", []),
    "Preload.TRANSPORTATION": ("float", []),
    "Preload.UTILITIES": ("float", []),
    "Preload.BASICCHEMICALS": ("float", []),
    "Preload.BEAUTYCARE": ("float", []),
    "Preload.COAL": ("float", []),
    "Preload.ENVIRONPROTECT": ("float", []),
    "Preload.PETROLEUM": ("float", []),
    "Preload.POWEREQUIP": ("float", []),
    "Preload.RETAILTRADE": ("float", []),
    "Preload.SOCIALSERVICES": ("float", []),
    "Preload.TEXTILEAPPAREL": ("float", []),
    "Preload.COUNTRY": ("float", []),

    # ============ Daily 字段 ============
    "Daily.Open": ("float", []),
    "Daily.PreClose": ("float", []),
    "Daily.UpperLimit": ("float", []),
    "Daily.LowerLimit": ("float", []),
    "Daily.OpenLimitTradeAmtBuy": ("float", []),
    "Daily.OpenLimitTradeAmtSell": ("float", []),
    "Daily.OpenLimitTradeTriggerOrdersBuy": ("float", []),
    "Daily.OpenLimitTradeTriggerOrdersSell": ("float", []),
    "Daily.OpenLimitTradeVolBuy": ("float", []),
    "Daily.OpenLimitTradeVolSell": ("float", []),
    "Daily.CloseAucTradeAmtBuy": ("float", []),
    "Daily.CloseAucTradeAmtSell": ("float", []),
    "Daily.CloseAucTradeTriggerOrdersBuy": ("float", []),
    "Daily.CloseAucTradeTriggerOrdersSell": ("float", []),
    "Daily.CloseAucTradeVolBuy": ("float", []),
    "Daily.CloseAucTradeVolSell": ("float", []),
    "Daily.TotalOutVol": ("float", []),
    "Daily.AdjFactor": ("float", []),


    # ============ Slice 基础字段 ============
    "Slice.Index": ("int", []),
    "Slice.ExpStart": ("int", []),
    "Slice.ExpEnd": ("int", []),
    "Slice.IsZT": ("bool", []),
    "Slice.IsDT": ("bool", []),
    "Slice.High": ("float", []),
    "Slice.Low": ("float", []),
    "Slice.CumHigh": ("float", []),
    "Slice.CumLow": ("float", []),
    "Slice.StartPrice": ("float", []),
    "Slice.LastPrice": ("float", []),
    "Slice.Ret": ("float", []),
    "Slice.NormalRet": ("float", []),
    "Slice.MidPrice": ("float", []),
    "Slice.AskPrice": ("vector", []),
    "Slice.BidPrice": ("vector", []),
    "Slice.AskVol": ("vector", []),
    "Slice.BidVol": ("vector", []),
    "Slice.SinceOpenVolatility": ("float", []),
    "Slice.SinceOpenSharpe": ("float", []),


    # ============ CS 字段 ============
    "CS.TradeVolResBuy": ("float", []),
    "CS.TradeVolResSell": ("float", []),
    "CS.TotalTradeVolRes": ("float", []),
    "CS.AggOrderVolResBuy": ("float", []),
    "CS.AggOrderVolResSell": ("float", []),
    "CS.TotalAggOrderVolRes": ("float", []),
    "CS.PasOrderVolResBuy": ("float", []),
    "CS.PasOrderVolResSell": ("float", []),
    "CS.TotalPasOrderVolRes": ("float", []),
    "CS.CancelVolResBuy": ("float", []),
    "CS.CancelVolResSell": ("float", []),
    "CS.TotalCancelVolRes": ("float", []),
    "CS.NormalRetRes": ("float", []),
    "CS.CumTradeVolResBuy": ("float", []),
    "CS.CumTradeVolResSell": ("float", []),
    "CS.CumTotalTradeVolRes": ("float", []),
    "CS.CumAggOrderVolResBuy": ("float", []),
    "CS.CumAggOrderVolResSell": ("float", []),
    "CS.CumTotalAggOrderVolRes": ("float", []),
    "CS.CumPasOrderVolResBuy": ("float", []),
    "CS.CumPasOrderVolResSell": ("float", []),
    "CS.CumTotalPasOrderVolRes": ("float", []),
    "CS.CumCancelVolResBuy": ("float", []),
    "CS.CumCancelVolResSell": ("float", []),
    "CS.CumTotalCancelVolRes": ("float", [])
}
result_dict_multi = {
        # ============ Slice 订单/成交统计字段 ============
    "Slice.____": ("float", [
        ["Cum", ""],
        ["AggOrder", "PasOrder", "Order", "Cancel", "Trade"],
        ["Cnt", "Amt", "Vol", "Aamt", "Vwap"],
        ["Buy", "Sell"]
    ]),  # Slice.[Cum](AggOrder|PasOrder|Order|Cancel|Trade)(Cnt|Amt|Vol|Aamt|Vwap)(Buy|Sell)
    
    "Slice.Total___": ("float", [
        ["Cum", ""],
        ["AggOrder", "PasOrder", "Order", "Cancel", "Trade"],
        ["Cnt", "Amt", "Vol", "Aamt", "Vwap"]
    ]),  # Slice.[Cum]Total(AggOrder|PasOrder|Order|Cancel|Trade)(Cnt|Amt|Vol|Aamt|Vwap)
    
    "Slice.__VolImbl": ("float", [
        ["Cum", ""],
        ["AggOrder", "PasOrder", "Order", "Cancel", "Trade"]
    ]),  # Slice.[Cum](AggOrder|PasOrder|Order|Cancel|Trade)VolImbl
    
    "Slice._Algo_Order__": ("float", [
        ["Cum", ""],
        ["Agg", "Pas"],
        ["Cnt", "Amt", "Vol", "Aamt", "Vwap"],
        ["Buy", "Sell"]
    ]),  # Slice.[Cum]Algo[Agg/Pas]Order(Cnt|Amt|Vol|Aamt|Vwap)(Buy|Sell)
    
    "Slice._TotalAlgo_Order_": ("float", [
        ["Cum", ""],
        ["Agg", "Pas"],
        ["Cnt", "Amt", "Vol", "Aamt", "Vwap"]
    ]),  # Slice.[Cum]TotalAlgo[Agg/Pas]Order(Cnt|Amt|Vol|Aamt|Vwap)
    
    "Slice._Algo_OrderVolImbl": ("float", [
        ["Cum", ""],
        ["Agg", "Pas"]
    ]),  # Slice.[Cum]Algo[Agg/Pas]OrderVolImbl
    
    "Slice._InstOrder__": ("float", [
        ["Cum", ""],
        ["Cnt", "Amt", "Vol", "Aamt", "Vwap"],
        ["Buy", "Sell"]
    ]),  # Slice.[Cum]InstOrder(Cnt|Amt|Vol|Aamt|Vwap)(Buy|Sell)

    # ============ Slice OB相关字段 ============
    "Slice.OBVol_": ("float", [["Buy", "Sell"]]),
    "Slice.OB10Vol_": ("float", [["Buy", "Sell"]]),
    "Slice.OBSupportVol_": ("float", [["Buy", "Sell"]]),
    "Slice.OB10SupportVol_": ("float", [["Buy", "Sell"]]),
    "Slice.OBVwap_": ("float", [["Buy", "Sell"]]),
    "Slice.OB10Vwap_": ("float", [["Buy", "Sell"]]),
    "Slice.OBSupportPrice_": ("float", [["Buy", "Sell"]]),
    "Slice.OB10SupportPrice_": ("float", [["Buy", "Sell"]]),
    "Slice.OBVolImbl": ("float", []),
    "Slice.OB10VolImbl": ("float", []),
    "Slice.OBSupportVolImbl": ("float", []),
    "Slice.OB10SupportVolImbl": ("float", []),
    "Slice.OBNetVolImbl": ("float", []),
    "Slice.OB10NetVolImbl": ("float", []),
    "Slice.OBNetVol_": ("float", [["Buy", "Sell"]]),
    "Slice.OB10NetVol_": ("float", [["Buy", "Sell"]]),
    "Slice.TotalOBNetVol": ("float", []),
    "Slice.TotalOB10NetVol": ("float", []),
    "Slice.OBAmt_": ("float", [["Buy", "Sell"]]),
    "Slice.OB10Amt_": ("float", [["Buy", "Sell"]]),
    "Slice.OBSupportAmt_": ("float", [["Buy", "Sell"]]),
    "Slice.OB10SupportAmt_": ("float", [["Buy", "Sell"]]),
    "Slice.OBNetAmt_": ("float", [["Buy", "Sell"]]),
    "Slice.OB10NetAmt_": ("float", [["Buy", "Sell"]]),
    "Slice.TotalOBNetAmt": ("float", []),
    "Slice.TotalOB10NetAmt": ("float", []),

    # ============ Slice 其他基础字段 ============
    "Slice.NetTDV": ("float", []),
    "Slice.CumNetTDV": ("float", []),
    "Slice.RetLiquidity": ("float", []),
    "Slice.CumBottom5PctPriceTradeVwapBuy": ("float", []),
    "Slice.CumBottom5PctPriceTradeVwapSell": ("float", []),
    "Slice.CumTop5PctPriceTradeVwapBuy": ("float", []),
    "Slice.CumTop5PctPriceTradeVwapSell": ("float", []),
    "Slice.CumBottom5PctVolTradeVwapBuy": ("float", []),
    "Slice.CumBottom5PctVolTradeVwapSell": ("float", []),
    "Slice.CumTop5PctVolTradeVwapBuy": ("float", []),
    "Slice.CumTop5PctVolTradeVwapSell": ("float", []),
    "Slice.CumTopVolTradeAvgPriceBuy": ("float", []),
    "Slice.CumTopVolTradeAvgPriceSell": ("float", []),
    "Slice.CumTotalBottom5PctPriceTradeVwap": ("float", []),
    "Slice.CumTotalTop5PctPriceTradeVwap": ("float", []),
    "Slice.CumTotalBottom5PctVolTradeVwap": ("float", []),
    "Slice.CumTotalTop5PctVolTradeVwap": ("float", []),
    "Slice.CumTotalTopVolTradeAvgPrice": ("float", []),

    # ============ Slice OrderAgg相关 ============
    "Slice.OrderAgg___": ("float", [
        ["Tda", "Tdc", "Tdv"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.OrderAgg(Tda|Tdc|Tdv)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.Order___": ("float", [
        ["Tda", "Tdc", "Tdv"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.Order(Tda|Tdc|Tdv)(Small|Mid|Big|Extra)(Buy|Sell)

    # ============ Slice Bias相关 ============
    "Slice.BiasAgg___": ("float", [
        ["Tda", "Tdc", "Tdv"],
        ["BB", "BS", "SB", "SS"],
        ["Buy", "Sell"]
    ]),  # Slice.BiasAgg(Tda|Tdc|Tdv)(BB|BS|SB|SS)(Buy|Sell)
    
    "Slice.Bias___": ("float", [
        ["Tda", "Tdc", "Tdv"],
        ["BB", "BS", "SB", "SS"],
        ["Buy", "Sell"]
    ]),  # Slice.Bias(Tda|Tdc|Tdv)(BB|BS|SB|SS)(Buy|Sell)

    # ============ Slice Cancel相关 ============
    "Slice.Cancel___": ("float", [
        ["Cnt", "Amt", "Vol"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.Cancel(Cnt|Amt|Vol)(Small|Mid|Big|Extra)(Buy|Sell)

    # ============ Slice Order相关 ============
    "Slice.OrderVol__": ("float", [
        ["Close", "Far"],
        ["Buy", "Sell"]
    ]),  # Slice.OrderVol(Close|Far)(Buy|Sell)
    
    "Slice.Order___": ("float", [
        ["Cnt", "Amt", "Vol"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.Order(Cnt|Amt|Vol)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.AggOrderAgg___": ("float", [
        ["Tdc", "Tdv"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.AggOrderAgg(Tdc|Tdv)(Small|Mid|Big|Extra)(Buy|Sell)

    # ============ Slice Tick相关 ============
    "Slice.TickOB10VolImbl_": ("vector", [["Mean", "Std"]]),  # Slice.TickOB10VolImbl(Mean|Std)
    "Slice.TickOB10SupportVolImblMean": ("float", []),
    "Slice.TickOB10NetVolImblMean": ("float", []),
    "Slice.RemainTickCntRatio_": ("float", [["Buy", "Sell"]]),
    "Slice.BidAskSpanDiffCatRatio": ("float", []),
    "Slice.PasOrderHalfSpreadVol_": ("float", [["Buy", "Sell"]]),
    "Slice.PasOrderSpreadVol_": ("float", [["Buy", "Sell"]]),
    "Slice._PerLevel__": ("vector", [
        ["AggOrder", "PasOrder", "Cancel"],
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.(AggOrder|PasOrder|Cancel)PerLevel(Cnt|Amt|Vol)(Buy|Sell)
    
    "Slice.TickOB9BestPriceDiffImbl_": ("vector", [["Mean", "Std"]]),  # Slice.TickOB9BestPriceDiffImbl(Mean|Std)
    "Slice.TickOB10PriceDist_Imbl": ("float", [["Mean", "Std", "Skew"]]),  # Slice.TickOB10PriceDist(Mean|Std|Skew)Imbl

    # ============ Slice CumSpan相关 ============
    "Slice.CumOrderReportToFirstFillSpan___": ("float", [
        ["Mean", "Std", "Skew", "Kurt"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumOrderReportToFirstFillSpan(Mean|Std|Skew|Kurt)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.CumOrderReportToCompleteFillSpan___": ("float", [
        ["Mean", "Std", "Skew", "Kurt"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumOrderReportToCompleteFillSpan(Mean|Std|Skew|Kurt)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.CumOrderFirstFillToCompleteFillSpan___": ("float", [
        ["Mean", "Std", "Skew", "Kurt"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumOrderFirstFillToCompleteFillSpan(Mean|Std|Skew|Kurt)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.CumOrderReportToFirstFillSpan25Quantiles__": ("vector", [
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumOrderReportToFirstFillSpan25Quantiles(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.CumOrderReportToCompleteFillSpan25Quantiles__": ("vector", [
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumOrderReportToCompleteFillSpan25Quantiles(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.CumOrderFirstFillToCompleteFillSpan25Quantiles__": ("vector", [
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumOrderFirstFillToCompleteFillSpan25Quantiles(Small|Mid|Big|Extra)(Buy|Sell)

    # ============ Slice Cum量价分位数 ============
    "Slice.Cum__25Quantiles_": ("vector", [
        ["Order", "Cancel", "Trade"],
        ["Vol", "Amt"],
        ["Buy", "Sell"]
    ]),  # Slice.Cum(Order|Cancel|Trade)(Vol|Amt)25Quantiles(Buy|Sell)

    # ============ Slice CumAdjacentSpan相关 ============
    "Slice.CumAdjacent_Span___": ("float", [
        ["AggOrder", "PasOrder", "Cancel", "Trade"],
        ["Mean", "Std", "Skew", "Kurt"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumAdjacent(AggOrder|PasOrder|Cancel|Trade)Span(Mean|Std|Skew|Kurt)(Small|Mid|Big|Extra)(Buy|Sell)

    # ============ Slice 撤单相关 ============
    "Slice._CancelledOrder___": ("float", [
        ["Partial", "Full"],
        ["Cnt", "Amt", "Vol"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.(Partial|Full)CancelledOrder(Cnt|Amt|Vol)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice._BaitOrder___": ("float", [
        ["Cum", ""],
        ["Cnt", "Amt", "Vol"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.[Cum]BaitOrder(Cnt|Amt|Vol)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.CumCancelled_Order___": ("float", [
        ["L2", "L3"],
        ["Cnt", "Amt", "Vol"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumCancelled(L2|L3)Order(Cnt|Amt|Vol)(Small|Mid|Big|Extra)(Buy|Sell)

    # ============ Slice RegretFear相关 ============
    "Slice._RegretFear__": ("float", [
        ["Cum", ""],
        ["Tdc", "Tda", "Tdv"],
        ["Buy", "Sell"]
    ]),  # Slice.[Cum]RegretFear(Tdc|Tda|Tdv)(Buy|Sell)

    # ============ Slice LongFullFillOrder相关 ============
    "Slice.CumLongFullFillOrder__": ("float", [
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.CumLongFullFillOrder(Cnt|Amt|Vol)(Buy|Sell)

    # ============ Slice OnceFullFill相关 ============
    "Slice.CumOnceFullFill_Order___": ("float", [
        ["Agg", "Pas"],
        ["Cnt", "Amt", "Vol"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.CumOnceFullFill(Agg|Pas)Order(Cnt|Amt|Vol)(Small|Mid|Big|Extra)(Buy|Sell)

    # ============ Slice TradePvDstr相关 ============
    "Slice.TradePvDstr__": ("float", [
        ["Mean", "Std", "Skew", "Kurt"],
        ["Buy", "Sell"]
    ]),  # Slice.TradePvDstr(Mean|Std|Skew|Kurt)(Buy|Sell)
    
    "Slice.TotalTradePvDstr_": ("float", [["Mean", "Std", "Skew", "Kurt"]]),  # Slice.TotalTradePvDstr(Mean|Std|Skew|Kurt)
    
    "Slice.TdvProbDstrMean__": ("float", [
        ["Mean", "Std", "Skew", "Kurt"],
        ["Buy", "Sell"]
    ]),  # Slice.TdvProbDstrMean(Mean|Std|Skew|Kurt)(Buy|Sell)
    
    "Slice.TotalTdvProbDstrMean_": ("float", [["Mean", "Std", "Skew", "Kurt"]]),  # Slice.TotalTdvProbDstrMean(Mean|Std|Skew|Kurt)

    # ============ Slice ProfitTrade相关 ============
    "Slice.ProfitTrade___": ("float", [
        ["Cnt", "Vol", "Amt"],
        ["Close", "Vwap"],
        ["Buy", "Sell"]
    ]),  # Slice.ProfitTrade(Cnt|Vol|Amt)Vs(Close|Vwap)(Buy|Sell)
    
    "Slice.LossTrade___": ("float", [
        ["Cnt", "Vol", "Amt"],
        ["Close", "Vwap"],
        ["Buy", "Sell"]
    ]),  # Slice.LossTrade(Cnt|Vol|Amt)Vs(Close|Vwap)(Buy|Sell)

    # ============ Slice Tdp相关 ============
    "Slice._20PctTdp__": ("float", [
        ["Top", "Bottom"],
        ["Tdc", "Tda", "Tdv"],
        ["Buy", "Sell"]
    ]),  # Slice.(Top|Bottom)20PctTdp(Tdc|Tda|Tdv)(Buy|Sell)
    
    "Slice.Total_20PctTdp_": ("float", [
        ["Top", "Bottom"],
        ["Tdc", "Tda", "Tdv"]
    ]),  # Slice.Total(Top|Bottom)20PctTdp(Tdc|Tda|Tdv)
    
    "Slice._20PctTdpPvDstr__": ("float", [
        ["Top", "Bottom"],
        ["Mean", "Std", "Skew", "Kurt"],
        ["Buy", "Sell"]
    ]),  # Slice.(Top|Bottom)20PctTdpPvDstr(Mean|Std|Skew|Kurt)(Buy|Sell)
    
    "Slice.Total_20PctTdpPvDstr_": ("float", [
        ["Top", "Bottom"],
        ["Mean", "Std", "Skew", "Kurt"]
    ]),  # Slice.Total(Top|Bottom)20PctTdpPvDstr(Mean|Std|Skew|Kurt)
    
    "Slice._20PctTdpTdvProbDstr__": ("float", [
        ["Top", "Bottom"],
        ["Mean", "Std", "Skew", "Kurt"],
        ["Buy", "Sell"]
    ]),  # Slice.(Top|Bottom)20PctTdpTdvProbDstr(Mean|Std|Skew|Kurt)(Buy|Sell)
    
    "Slice.Total_20PctTdpTdvProbDstr_": ("float", [
        ["Top", "Bottom"],
        ["Mean", "Std", "Skew", "Kurt"]
    ]),  # Slice.Total(Top|Bottom)20PctTdpTdvProbDstr(Mean|Std|Skew|Kurt)

    # ============ Slice PriceLifespan相关 ============
    "Slice._PriceLifespan_": ("float", [
        ["Top", "Bottom"],
        ["Buy", "Sell"]
    ]),  # Slice.(Top|Bottom)PriceLifespan(Buy|Sell)
    
    "Slice.Total_PriceLifespan": ("float", [["Top", "Bottom"]]),  # Slice.Total(Top|Bottom)PriceLifespan

    # ============ Slice PartialSpan1by3相关 ============
    "Slice.PartialSpan1by3___": ("vector", [
        ["AggOrder", "PasOrder", "Cancel", "Trade"],
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.PartialSpan1by3(AggOrder|PasOrder|Cancel|Trade)(Cnt|Amt|Vol)(Buy|Sell)

    # ============ Slice GiveupCancel相关 ============
    "Slice.GiveupCancel__": ("float", [
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.GiveupCancel(Cnt|Amt|Vol)(Buy|Sell)
    
    "Slice.ChaseCancel__": ("float", [
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.ChaseCancel(Cnt|Amt|Vol)(Buy|Sell)

    # ============ Slice Midprice相关 ============
    "Slice.Midprice__": ("float", [
        ["Up", "Down"],
        ["Cnt", "Size", "NormedSpan"]
    ]),  # Slice.Midprice(Up|Down)(Cnt|Size|NormedSpan)
    
    "Slice._CauseMidp__": ("float", [
        ["AggOrder", "PasOrder", "Cancel"],
        ["Up", "Down"],
        ["Cnt", "Amt", "Vol"]
    ]),  # Slice.(AggOrder|PasOrder|Cancel)CauseMidp(Up|Down)(Cnt|Amt|Vol)
    
    "Slice.MidpriceStay___": ("float", [
        ["AggOrder", "PasOrder", "Cancel"],
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.MidpriceStay(AggOrder|PasOrder|Cancel)(Cnt|Amt|Vol)(Buy|Sell)
    
    "Slice._CauseMidp_GapMean": ("float", [
        ["AggOrder", "PasOrder", "Order", "Cancel"],
        ["Up", "Down"]
    ]),  # Slice.(AggOrder|PasOrder|Order|Cancel)CauseMidp(Up|Down)GapMean

    # ============ Slice TradeAmtOutStdRatio相关 ============
    "Slice.TradeAmtOutStdRatio_": ("vector", [["Buy", "Sell"]]),  # Slice.TradeAmtOutStdRatio(Buy|Sell)

    # ============ Slice CancelOnSupportLevel相关 ============
    "Slice.CancelOnSupportLevel__": ("float", [
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.CancelOnSupportLevel(Cnt|Amt|Vol)(Buy|Sell)

    # ============ Slice Ob相关 ============
    "Slice.Ob_Imbl_": ("float", [
        ["Mouth", "10Vw"],
        ["Slope", "Intercept"]
    ]),  # Slice.Ob(Mouth|10Vw)Imbl(Slope|Intercept)
    
    "Slice.TickLastprice_": ("float", [["Slope", "Intercept"]]),  # Slice.TickLastprice(Slope|Intercept)
    
    "Slice.TickTradeVol_": ("float", [["Slope", "Intercept"]]),  # Slice.TickTradeVol(Slope|Intercept)
    
    "Slice.ObMouthTwVolMean_": ("float", [["Buy", "Sell"]]),  # Slice.ObMouthTwVolMean(Buy|Sell)
    
    "Slice.ObMouthTwSquareVolMean_": ("float", [["Buy", "Sell"]]),  # Slice.ObMouthTwSquareVolMean(Buy|Sell)

    # ============ Slice Trade分类相关 ============
    "Slice._Trade__": ("float", [
        ["Round", "Odd", "Retail"],
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.(Round|Odd|Retail)Trade(Cnt|Amt|Vol)(Buy|Sell)

    # ============ Slice Price状态相关 ============
    "Slice.Price_NormedSpan": ("float", [["Up", "Down", "Stay"]]),  # Slice.Price(Up|Down|Stay)NormedSpan
    
    "Slice.Price_Trade_": ("float", [
        ["Up", "Down", "Stay"],
        ["Cnt", "Amt", "Vol"]
    ]),  # Slice.Price(Up|Down|Stay)Trade(Cnt|Amt|Vol)
    
    "Slice.LongestPrice_Trade_": ("float", [
        ["Up", "Down"],
        ["Cnt", "Amt", "Vol"]
    ]),  # Slice.LongestPrice(Up|Down)Trade(Cnt|Amt|Vol)
    
    "Slice.LongestPrice__": ("float", [
        ["Up", "Down"],
        ["PasOrderCnt", "Span", "Size"]
    ]),  # Slice.LongestPrice(Up|Down)(PasOrderCnt|Span|Size)

    # ============ Slice Ret相关 ============
    "Slice._TradeRetSquareSum": ("float", [["Pos", "Neg"]]),  # Slice.(Pos|Neg)TradeRetSquareSum

    # ============ Slice Cancel寿命相关 ============
    "Slice._Cancel_Lifespan_": ("float", [
        ["Optimal", ""],
        ["Avg", "Vwa"],
        ["Buy", "Sell"]
    ]),  # Slice.[Optimal]Cancel(Avg|Vwa)Lifespan(Buy|Sell)

    # ============ Slice D1Rel相关 ============
    "Slice.D1Rel_Order___": ("float", [
        ["Order", "Cancel"],
        ["Cnt", "Amt", "Vol"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.D1Rel(Order|Cancel)(Cnt|Amt|Vol)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.OrderFilledPartMax__": ("float", [
        ["Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.OrderFilledPartMax(Amt|Vol)(Buy|Sell)
    
    "Slice.MaxFilledVolOrderPrice_": ("float", [["Buy", "Sell"]]),  # Slice.MaxFilledVolOrderPrice(Buy|Sell)
    
    "Slice.FilledOrder__": ("float", [
        ["Cnt", "Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.FilledOrder(Cnt|Amt|Vol)(Buy|Sell)
    
    "Slice.D1Rel__OrderFilled_": ("float", [
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"],
        ["Cnt", "Amt", "Vol", "Orders"]
    ]),  # Slice.D1Rel(Small|Mid|Big|Extra)(Buy|Sell)OrderFilled(Cnt|Amt|Vol|Orders)
    
    "Slice.D1Rel__FilledOrderRptVwap": ("float", [
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.D1Rel(Small|Mid|Big|Extra)(Buy|Sell)FilledOrderRptVwap
    
    "Slice.MaxOrderFilled_InTrade_": ("float", [
        ["Amt", "Vol"],
        ["Buy", "Sell"]
    ]),  # Slice.MaxOrderFilled(Amt|Vol)InTrade(Buy|Sell)
    
    "Slice.MaxFilledVolOrderPriceInTrade_": ("float", [["Buy", "Sell"]]),  # Slice.MaxFilledVolOrderPriceInTrade(Buy|Sell)
    
    "Slice.FilledOrderRptVwapInTrade_": ("float", [["Buy", "Sell"]]),  # Slice.FilledOrderRptVwapInTrade(Buy|Sell)
    
    "Slice.FilledOrderCntInTrade_": ("float", [["Buy", "Sell"]]),  # Slice.FilledOrderCntInTrade(Buy|Sell)
    
    "Slice.D1Rel_OrderFilled_In_Trade": ("float", [
        ["Small", "Mid", "Big", "Extra"],
        ["Cnt", "Amt", "Vol", "Orders"],
        ["Buy", "Sell"]
    ]),  # Slice.D1Rel(Small|Mid|Big|Extra)OrderFilled(Cnt|Amt|Vol|Orders)In(Buy|Sell)Trade
    
    "Slice.D1Rel_FilledOrderRptVwapIn_Trade": ("float", [
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.D1Rel(Small|Mid|Big|Extra)FilledOrderRptVwapIn(Buy|Sell)Trade
    
    "Slice.Trade_InTrade_": ("float", [
        ["Cnt", "Amt", "Vol"],
        ["Up", "Down"]
    ]),  # Slice.Trade(Cnt|Amt|Vol)InTrade(Up|Down)
    
    "Slice.MaxOrderFilled_InTrade_": ("float", [
        ["Amt", "Vol"],
        ["Up", "Down"]
    ]),  # Slice.MaxOrderFilled(Amt|Vol)InTrade(Up|Down)
    
    "Slice.MaxFilledVolOrderPriceInTrade_": ("float", [["Up", "Down"]]),  # Slice.MaxFilledVolOrderPriceInTrade(Up|Down)
    
    "Slice.FilledOrderRptVwapInTrade_": ("float", [["Up", "Down"]]),  # Slice.FilledOrderRptVwapInTrade(Up|Down)
    
    "Slice.FilledOrderCntInTrade_": ("float", [["Up", "Down"]]),  # Slice.FilledOrderCntInTrade(Up|Down)
    
    "Slice.D1Rel_OrderFilled_In_Trade": ("float", [
        ["Small", "Mid", "Big", "Extra"],
        ["Up", "Down"],
        ["Cnt", "Amt", "Vol", "Orders"]
    ]),  # Slice.D1Rel(Small|Mid|Big|Extra)OrderFilled(Cnt|Amt|Vol|Orders)In(Up|Down)Trade
    
    "Slice.D1Rel_FilledOrderRptVwapIn_Trade": ("float", [
        ["Small", "Mid", "Big", "Extra"],
        ["Up", "Down"]
    ]),  # Slice.D1Rel(Small|Mid|Big|Extra)FilledOrderRptVwapIn(Up|Down)Trade

    # ============ Slice D1RelOB相关 ============
    "Slice.D1RelOBLevel___": ("float", [
        ["PriceMean", "Cnt"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.D1RelOBLevel(PriceMean|Cnt)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.D1RelOB___": ("float", [
        ["Cnt", "Amt", "Vol", "Vwap"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.D1RelOB(Cnt|Amt|Vol|Vwap)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.D1RelOBLevelVol___": ("float", [
        ["Std", "Skew", "Kurt", "HHI"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.D1RelOBLevelVol(Std|Skew|Kurt|HHI)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.OBLevel__": ("float", [
        ["PriceMean", "Cnt"],
        ["Buy", "Sell"]
    ]),  # Slice.OBLevel(PriceMean|Cnt)(Buy|Sell)
    
    "Slice.OBCnt_": ("float", [["Buy", "Sell"]]),  # Slice.OBCnt(Buy|Sell)
    
    "Slice.OBLevelVol__": ("float", [
        ["Std", "Skew", "Kurt", "HHI"],
        ["Buy", "Sell"]
    ]),  # Slice.OBLevelVol(Std|Skew|Kurt|HHI)(Buy|Sell)

    # ============ Slice D1RelTB相关 ============
    "Slice.D1RelTB___": ("float", [
        ["Cnt", "Amt", "Vol", "Vwap", "OrderCnt"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.D1RelTB(Cnt|Amt|Vol|Vwap|OrderCnt)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.D1RelTBLevelVol___": ("float", [
        ["Std", "Skew", "Kurt", "HHI"],
        ["Small", "Mid", "Big", "Extra"],
        ["Buy", "Sell"]
    ]),  # Slice.D1RelTBLevelVol(Std|Skew|Kurt|HHI)(Small|Mid|Big|Extra)(Buy|Sell)
    
    "Slice.TB__": ("float", [
        ["Cnt", "Amt", "Vol", "Vwap", "OrderCnt"],
        ["Buy", "Sell"]
    ]),  # Slice.TB(Cnt|Amt|Vol|Vwap|OrderCnt)(Buy|Sell)
    
    "Slice.TBLevelVol__": ("float", [
        ["Std", "Skew", "Kurt", "HHI"],
        ["Buy", "Sell"]
    ]),  # Slice.TBLevelVol(Std|Skew|Kurt|HHI)(Buy|Sell)

    # ============ Slice EventMask相关 ============
    "Slice.EventMask_": ("float", [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]]),  # Slice.EventMask(1~26)
    "Daily.OrderAmtThres1_": ("float", [["Buy", "Sell"]]),
    "Daily.OrderAmtThres2_": ("float", [["Buy", "Sell"]]),
    "Daily.OrderAmtThres3_": ("float", [["Buy", "Sell"]]),
}