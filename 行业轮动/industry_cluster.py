#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-05-10 17:14
# @Author : lichangheng

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

dfs = pd.read_excel('./industry.xlsx', sheet_name=['week', 'month'])
df1 = dfs.get('week').set_index('行业名称').drop(['指数代码'], axis=1)

# 将需要聚类的数据转换为 NumPy 数组
X = df1.values

# 计算数据的距离矩阵
dist_matrix = linkage(X, method='ward')

# 绘制树状图
dendrogram(dist_matrix)

# 自动确定聚类数
last = dist_matrix[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=100)
ax[0].plot(idxs, last_rev)

acceleration = np.diff(last, 2)
acceleration_rev = acceleration[::-1]
ax[1].plot(idxs[:-2] + 1, acceleration_rev)

k = acceleration_rev.argmax() + 5  # 聚类数为加速度最大的位置加 2

# 将数据分配到不同的聚类中，并输出每个数据点所属的聚类编号
clusters = fcluster(dist_matrix, k, criterion='maxclust')
s = pd.DataFrame(list(zip(df1.index,clusters)),columns=['index','cluster']).groupby('cluster')
print(list(s))
# 显示图形
plt.show()

