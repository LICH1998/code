#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-03-14 15:53
# @Author : lichangheng

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost_ts import *
from tqdm import *


def max_back(prices):
    max_price = 0
    max_d = 0
    for price in prices[1:]:
        if price > max_price:
            max_price = price
        else:
            drawdown = (max_price - price) / max_price
            max_d = max(max_d, drawdown)
    return max_d


# 回测函数
def backtest(df, seq_close):
    position = 0
    ini_cash = 100
    cash = 100
    signal = []
    total_value = []
    for i in tqdm(range(len(df))):
        data = seq_close[i:200 + i]
        s_sig = data_process(data)
        if s_sig and cash > df.close.iloc[i]:
            signal.append(1)
            position += round(cash / df.close.iloc[i])
            cash -= round(cash / df.close.iloc[i]) * df.close.iloc[i]
            total_value.append(cash + position * df.close.iloc[i])
        elif not s_sig and position:
            signal.append(-1)
            cash += df.close.iloc[i] * position
            position = 0
            total_value.append(cash + position * df.close.iloc[i])
        else:
            signal.append(0)
            total_value.append(cash + position * df.close.iloc[i])
    df['signal'] = signal
    buy_scatter = df[df['signal'] == 1].index.tolist()
    sell_scatter = df[df['signal'] == -1].index.tolist()
    # 计算每天的总值
    df['total_value'] = total_value
    # 计算每天的收益率
    df['returns'] = df['total_value'].pct_change()
    # 计算累计收益率
    df['cum_returns'] = (1 + df['returns']).cumprod()
    # 计算最大回撤率
    max_drawdown = max_back(np.array(df['cum_returns']))
    # 计算年化收益率和夏普比率
    n_years = len(df) / 252
    annualized_returns = (df['cum_returns'][-1]) ** (1 / n_years) - 1
    annualized_volatility = df['returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_returns / annualized_volatility
    # 绘制收益曲线和RSI曲线
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=100)
    ax[0].plot(df.index, df['cum_returns'], label='Cumulative Returns', c='black')
    ax[0].scatter(buy_scatter, df['cum_returns'][buy_scatter], color='red', s=20, alpha=1, label='Buy')
    ax[0].scatter(sell_scatter, df['cum_returns'][sell_scatter], color='blue', s=20, alpha=1, label='Sell', marker='*')
    ax[0].legend(["Cumulative Returns", 'Buy', 'Sell'], loc="upper left", fontsize=11)
    ax[1].plot(df.index, df['close'], label='CLOSE')
    ax[1].legend(["CLOSE"], loc="upper left", fontsize=11)
    plt.show()
    return annualized_returns, max_drawdown, sharpe_ratio


# 加载股票数据
df = pd.read_csv('./data1.csv', index_col='bob', parse_dates=True)
# 传入close变化率
seq_close = np.array(df['close'].pct_change())
df = df.iloc[199:]
# 进行回测
annualized_returns, max_drawdown, sharpe_ratio = backtest(df, seq_close)
# 打印结果
print('Annualized Returns: {:.2f}%'.format(annualized_returns * 100))
print('Max Drawdown: {:.2f}%'.format(-max_drawdown * 100))
print('Sharpe Ratio: {:.2f}'.format(sharpe_ratio))
