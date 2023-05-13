#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-05-12 23:57
# @Author : lichangheng

import pandas as pd
import numpy as np
import math


# 计算ATR指标
def calc_atr(close_prices, high_prices, low_prices, n=1):
    tr = pd.DataFrame(index=close_prices.index, columns=['TR'])
    tr['p1'] = abs(high_prices - low_prices)
    tr['p2'] = abs(high_prices - close_prices.shift())
    tr['p3'] = abs(low_prices - close_prices.shift())
    tr['TR'] = tr[['p1', 'p2', 'p3']].max(axis=1)
    atr = tr['TR'].pct_change().rolling(n).mean()
    return atr


# 计算历史波动率指标
def calc_hv(close_prices, n=1):
    log_ret = np.log(close_prices / close_prices.shift())
    hv = log_ret.rolling(n).std() * math.sqrt(252)
    return hv


# 计算股价波动率
def calc_volatility(close_prices, high_prices, low_prices, volume, turnover):
    atr = calc_atr(close_prices, high_prices, low_prices)
    hv = calc_hv(close_prices)
    vol = pd.DataFrame(index=close_prices.index, columns=['Volatility'])
    vol['ATR'] = atr / close_prices
    vol['HV'] = hv / close_prices
    vol['Volume'] = volume / volume.mean()
    vol['Turnover'] = turnover / turnover.mean()
    vol['Volatility'] = vol.mean(axis=1)
    return vol['Volatility'].mean()
