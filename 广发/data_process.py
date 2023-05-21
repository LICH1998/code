#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-05-19 22:16
# @Author : lichangheng

import pandas as pd
import numpy as np
import math
import os
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


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


def run():
    filePath = './stock_component'
    fileList = os.listdir(filePath)
    data_list = []
    column = []
    for i in tqdm(range(len(fileList))):
        # 读取数据
        file = os.path.join(filePath, fileList[i])
        data = pd.read_csv(file, encoding='gb18030').set_index('日期').iloc[:, :-1]
        data.columns = ['code', 'name', 'high', 'low', 'close', 'volume', 'amount', 'chg', 'turn', 'm_value',
                        's_capital', 'pe', 'pb', 'ps', 'pm']
        # 数据预处理
        data.iloc[:, -2:] = data.iloc[:, -2:].replace('--', 0)
        data = data.replace('--', np.NaN).dropna(how='any', axis=0)
        data.iloc[:, 2:] = data.iloc[:, 2:].astype(float)
        # 生成波动性指标
        data['vol'] = calc_volatility(data['close'], data['high'], data['low'], data['volume'], data['turn'])
        # 归一化(交易指标)
        data.iloc[:, 2:7] = data.iloc[:, 2:7].apply(lambda x: (x - x.min()) / (x.max() - x.min())).fillna(0)
        res = [data.code[0],data.name[0]]
        res.extend(data.mean().values)
        res.append(data['turn'].std())
        data_list.append(res)
        column = list(data.columns)
        column.append('turn_std')

    # 存为文件
    data_list = pd.DataFrame(data_list, columns=column).to_csv(
        'data_total.csv', index=False)


if __name__ == '__main__':
    run()
