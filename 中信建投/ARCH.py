#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-05-12 21:29
# @Author : lichangheng

import numpy as np
import pandas as pd
from arch import arch_model


def arch_func(data):
    returns = 100*data['close'].pct_change().dropna()  # 计算收益率并移除缺失值
    am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')  # 创建GARCH模型
    model = am.fit(disp='off')  # 拟合模型
    forecasts = model.forecast(horizon=10)
    var = forecasts.residual_variance.values[-1,0]  # 获取波动率
    return var


# data = pd.read_csv('./date_total/301157.SZ.CSV', encoding='gb18030').drop(columns='Unnamed: 9').fillna(method='ffill')
# data.columns = ['code', 'name', 'date', 'high', 'low', 'close', 'volume', 'chg', 'turn']
# a = arch_func(data[:5])
# print(a)
