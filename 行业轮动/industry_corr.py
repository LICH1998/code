#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-05-10 14:36
# @Author : lichangheng

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dfs = pd.read_excel('./industry.xlsx', sheet_name=['week', 'month'])
df1 = dfs.get('week').set_index('行业名称').drop(['指数代码'], axis=1).T
a = df1.corr()
print(a)
res1 = []
res2 = []
for i in a.index:
    l1 = a[i].nlargest(2).min()
    l2 = a[i].nlargest(2).idxmin()
    l3 = a[i].nlargest(3).min()
    l4 = a[i].nlargest(3).idxmin()
    res1.append((i,l1,l2))
    res2.append((i,l3,l4))
print("第一类相关行业: ",res1)
print("第二类相关行业: ",res2)

# 画热力图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决无法显示符号的问题
sns.set(font='SimHei', font_scale=0.8)        # 解决Seaborn中文显示问题
ax = sns.heatmap(a, annot=False,linewidths=0.1,linecolor="grey", cmap='RdBu_r')
ax.plot()
plt.show()

