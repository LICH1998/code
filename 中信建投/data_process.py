#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-05-12 21:29
# @Author : lichangheng

import pandas as pd
import numpy as py
from ARCH import *
from volatility import *
import os
import math
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题


def init_param():
    # 回测参数 ----------
    param = dict()
    param['method'] = 'arch'  # arch or volatility or std
    param['filepath'] = './data_total'  # path-----data_total or  data_total_cmp or data(仅涨跌幅分布)
    param['plot'] = ['./data_total', './data_total_cmp']
    param['percentile'] = np.arange(0, 1.1, 0.1)  # 分位数

    return param


def run(param):
    filePath = param['filepath']
    fileList = os.listdir(filePath)
    # 存放波动率
    vlt_pre = []
    vlt_aft = []
    for file in fileList:
        # 读取每支股票数据
        file_ = os.path.join(filePath, file)
        data = pd.read_csv(file_, encoding='gb18030').drop(columns='Unnamed: 9').fillna(method='ffill')
        data.columns = ['code', 'name', 'date', 'high', 'low', 'close', 'volume', 'chg', 'turn']
        data.set_index('date', inplace=True)
        # 划分前五条数据和后续数据
        data_pre = data.iloc[:5]
        data_aft = data.iloc[5:10]
        if param['method'] is 'arch':
            vlt_pre.append(arch_func(data_pre))
            vlt_aft.append(arch_func(data_aft))
        elif param['method'] is 'volatility':
            vlt_pre.append(calc_volatility(data_pre['close'], data_pre['high'], data_pre['low'], data_pre['volume'],
                                           data_pre['turn']))
            vlt_aft.append(calc_volatility(data_aft['close'], data_aft['high'], data_aft['low'], data_aft['volume'],
                                           data_aft['turn']))
        elif param['method'] is 'std':
            vlt_pre.append(data_pre['close'].pct_change().fillna(0).std())
            vlt_aft.append(data_aft['close'].pct_change().fillna(0).std())
        else:
            pass
    res = pd.DataFrame(list(zip(vlt_pre, vlt_aft)), columns=['vlt_pre', 'vlt_aft'])
    print(res.describe())
    # 进行假设检验
    out = stats.ttest_ind(list(res['vlt_pre']), list(res['vlt_aft']), equal_var=False)
    print("t统计量的值为 {} ,P值为 {} ".format(list(out)[0], list(out)[1]))
    if list(out)[1] < 0.05:
        print('拒绝原假设')
    else:
        print('接受原假设')


def res_plot(param):
    index = [i + 1 for i in range(10)]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=100)

    for _, filePath in enumerate(param['plot']):
        fileList = os.listdir(filePath)
        pct = []
        for file in fileList:
            # 读取每支股票数据
            file_ = os.path.join(filePath, file)
            data = pd.read_csv(file_, encoding='gb18030').drop(columns='Unnamed: 9').fillna(method='ffill')
            data.columns = ['code', 'name', 'date', 'high', 'low', 'close', 'volume', 'chg', 'turn']
            data.set_index('date', inplace=True)
            pct.append(list(data.iloc[:10]['close'].pct_change().fillna(0)))
        # 股价涨跌幅可视化
        pct = pd.DataFrame(pct, columns=index).mean(0)
        if _:
            ax.plot(index, pct, label='注册制前', c='red', linewidth=3.0)
        else:
            ax.plot(index, pct, label='注册制后', c='blue', linewidth=3.0)
    ax.legend(["注册制前", '注册制后'], loc="upper left", fontsize=11)
    ax.vlines([5], -0.04, 0.01, linestyles='dashed', colors='red')
    plt.xticks(index)
    plt.xlabel('时间')
    plt.ylabel('涨跌幅')
    plt.title('Pct_change Graph', fontsize='xx-large', fontweight='heavy')
    plt.show()


def calc_percentile(param):
    percentiles = param['percentile']
    filePath = param['filepath']
    fileList = os.listdir(filePath)
    # 存放涨跌幅数据
    chg = []
    for file in fileList:
        # 读取每支股票数据
        file_ = os.path.join(filePath, file)
        data = pd.read_csv(file_, encoding='gb18030').drop(columns='Unnamed: 9').fillna(method='ffill')
        if len(data)<6:
            continue
        data.columns = ['code', 'name', 'date', 'high', 'low', 'close', 'volume', 'chg', 'turn']
        data.set_index('date', inplace=True)
        # 前五条数据
        chg.extend(list(data.iloc[:6]['close'].pct_change()[1:]))
    stock_percentiles = np.percentile(chg, percentiles * 100)
    print('分位数： ', stock_percentiles)


param = init_param()
# 计算波动性
run(param)
# 可视化
res_plot(param)
# 计算分位数
calc_percentile(param)

