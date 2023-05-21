#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-05-20 17:14
# @Author : lichangheng

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

df = pd.read_csv('./data_total.csv').set_index('code')

# 把换手率范围存为文件
df['turn_down'] = df['turn'] - 1.96 * df['turn_std']
df['turn_top'] = df['turn'] + 1.96 * df['turn_std']
out = df.iloc[:, -2:].reset_index()
out.to_csv('./turn.csv', index=False)

# 将需要聚类的数据转换为 NumPy 数组
X = copy.deepcopy(df).drop(columns=['name', 's_capital', 'm_value', 'pe', 'pb', 'ps', 'pm', 'turn_std', 'turn_down',
                                    'turn_top']).values

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

k = acceleration_rev.argmax() + 4  # 聚类数为加速度最大的位置加 3

# 将数据分配到不同的聚类中，并输出每个数据点所属的聚类编号
clusters = fcluster(dist_matrix, k, criterion='maxclust')
df['cluster'] = clusters

# 把分类结果存为文件
cluster_res = pd.DataFrame(0,index=df.index,columns=['name','label']).reset_index()
cluster_res['name'] = df['name'].values
cluster_res['label'] = df['cluster'].values
cluster_res['label'] = cluster_res['label'].map({1:'A',2:'B',3:'C',4:'D'})
cluster_res.to_csv('./cluster_res.csv',index=False)

s = df.groupby('cluster').agg(
    {'name': 'count', 'pe': 'mean', 'chg': 'mean', 'vol': 'mean', 'turn': 'mean', 'amount': 'mean',
     'm_value': 'mean', 'pb': 'mean', 'ps': 'mean', 'high': 'mean', 'low': 'mean', 'close': 'mean', 'volume': 'mean'})
s = s.rename(columns={'name': 'count'})
s.index = ['A','B','C','D']
s.to_csv('./res.csv',index=False)

print(s)
# 显示图形
plt.show()

# 聚类可视化
# 使用TSNE进行数据降维并展示聚类结果
from sklearn.manifold import TSNE

tsne = TSNE()
tsne.fit_transform(X)  # 进行数据降维
# tsne.embedding_可以获得降维后的数据
tsn = pd.DataFrame(tsne.embedding_, index=df.index)  # 转换数据格式
# 不同类别用不同颜色和样式绘图
type_ = ['A', 'B', 'C', 'D']
color_style = ['r+', 'go', 'b*', 'kx']
for i in range(4):
    d = tsn[df[u'cluster'] == i + 1]
    # dataframe格式的数据经过切片之后可以通过d[i]来得到第i列数据
    plt.plot(d[0], d[1], color_style[i], label=type_[i])
plt.legend()
plt.title('聚类可视化结果')
plt.show()
