#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-03-14 11:33
# @Author : lichangheng
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tqdm
import joblib
import copy
import random
import os
import matplotlib.pyplot as plt
import math

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import extract_features, EfficientFCParameters, MinimalFCParameters

from xgboost.sklearn import XGBRegressor
import xgboost as xgb

from datetime import datetime

xgb.set_config(verbosity=0)

import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimHei']

input_window = 50
# 多步预测数
output_window = 1


def create_inout_sequences(input_data):
    inout_seq = []
    L = len(input_data)
    for i in range(L - input_window):
        train_seq = np.append(input_data[i:i + input_window][:-output_window], output_window * [0])
        train_label = input_data[i + input_window:i + input_window + output_window]
        inout_seq.append((train_seq, train_label))
    return np.array(inout_seq)


# 日频数据
# df = pd.read_csv('./dataset/csv/csv/002384.SZ_2022-12-12 08_45_06_2023-03-10 16_00_27.csv').fillna(
#     value=0)
# df['time_day'] = df['time'].map(lambda x: x[-8:])
# # 去除9:25之前和15:00之后的数据
# a = df[df['time_day']<'09:25:00'].index.tolist()
# a.extend(df[df['time_day']>'15:00:00'].index.tolist())
# df.drop(index=a,inplace=True)
# # 处理成datetime格式
# df['time'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
# df.reset_index(drop=True)

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=50)
# df['avgSellPrice'].plot(ax=axes, legend=True, color='blue',label='avgSellPrice')
# df['avgBuyPrice'].plot(ax=axes, legend=True, color='red',label='avgBuyPrice')
# plt.show()



def data_process(data):
    test_x = data[-input_window:].reshape(1,-1)
    a = create_inout_sequences(data)
    train_x = np.stack([item[0] for item in a])
    train_y = np.stack([item[1] for item in a])

    # 定义模型
    bst = xgb.XGBRegressor(max_depth=5,  # 每一棵树最大深度，默认6；
                           learning_rate=0.05,  # 学习率，每棵树的预测结果都要乘以这个学习率，默认0.3；
                           n_estimators=100,  # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
                           objective='reg:linear',  # 此默认参数与 XGBClassifier 不同
                           booster='gbtree',
                           # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认为gbtree
                           gamma=0,  # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                           min_child_weight=1,  # 可以理解为叶子节点最小样本数，默认1；
                           subsample=1,  # 训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
                           colsample_bytree=1,  # 每次拟合一棵树之前，决定使用多少个特征，参数默认1，取值范围(0, 1]。
                           reg_alpha=0,  # 默认为0，控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
                           reg_lambda=1,  # 默认为1，控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                           random_state=random.randint(1,10000))  # 随机种子
    bst.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='rmse', verbose=None)
    ans = bst.predict(test_x)
    if ans>0:
        return True
    else:
        return False

# # 数据测试
# data = np.array([math.sin(i) for i in range(0, 1000, 1)])
# data_process(data)