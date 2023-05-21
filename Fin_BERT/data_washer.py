# -*- coding: utf-8 -*-
"""
@author: HTSC
"""

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_csv(file_path, filtered_data, first):
    file_content = pd.read_csv(file_path)
    file_content.dropna(axis='index', how='any', inplace=True)  # 数据不全的字段都不要
    if first:
        first = False
        return file_content, first
    else:
        filtered_data = pd.concat([filtered_data, file_content], ignore_index=True)  # 把筛出来的数据拼接起来
        return filtered_data, first

def pre_text(text):
    '''
    将乱码、标点、字母和数字替换为空字符
    '''
    ignore_text = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
    ignore_text += '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~x000D\n0123456789 '
    ignore_text += 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if type(text) == str:
        return re.sub('[{}]'.format(ignore_text), '', text)
    else:
        return ''


data_dir = "../raw_data/WIND_FinancialNews"
file_list = os.listdir(data_dir)

first = True
filtered_data = None
for file in tqdm(file_list):
    publish_year = int(file[-10:-8])
    if 15 <= publish_year <= 17: # 拼接15-17年的所有文本数据
        filtered_data, first = read_csv(os.path.join(data_dir, file), filtered_data, first)
        print(file[-12:-4])

filtered_data.to_csv('../temp_data/WIND_FinancialNews_cleaned/filtered_data.csv', index=False, encoding='utf_8_sig')

# filtered_data = pd.read_csv('../data/WIND_FinancialNews_cleaned/filtered_data.csv')
filtered_data.rename(columns={'Unnamed: 0': 'INDEX'}, inplace=True)

# 筛选出与A股个股相关的新闻
filtered_data = filtered_data[filtered_data['MKTSENTIMENTS'].str.contains('0301:A股')]
filtered_data = filtered_data[filtered_data['MKTSENTIMENTS'].str.contains('03:公司')]

# 删除行业类新闻
filtered_data = filtered_data.drop(filtered_data[filtered_data['MKTSENTIMENTS'].str.contains('02:行业')].index)

# 删除标题中含有特定字符的新闻
filtered_data = filtered_data.drop(filtered_data[filtered_data['TITLE'].str.contains('走强')].index)
filtered_data = filtered_data.drop(filtered_data[filtered_data['TITLE'].str.contains('涨')].index)
filtered_data = filtered_data.drop(filtered_data[filtered_data['TITLE'].str.contains('跌')].index)
filtered_data = filtered_data.drop(filtered_data[filtered_data['TITLE'].str.contains('拉升')].index)
filtered_data = filtered_data.drop(filtered_data[filtered_data['TITLE'].str.contains('封板')].index)
filtered_data = filtered_data.drop(filtered_data[filtered_data['TITLE'].str.contains('快讯')].index)
filtered_data = filtered_data.drop(filtered_data[filtered_data['TITLE'].str.contains('正负面消息速览')].index)

# 将标题与内容合并
filtered_data.loc[:, 'CONTENT'] = filtered_data.loc[:, 'TITLE'] + '。' + filtered_data.loc[:, 'CONTENT']
del filtered_data['TITLE']

# 去掉无效字符
filtered_data.loc[:, 'CONTENT'] = filtered_data.loc[:, 'CONTENT'].apply(pre_text)

# 打情感标签
label_check = lambda x: 1 if x.count('正面') > x.count('负面') else 0
filtered_data.loc[:, 'MKTSENTIMENTS'] = filtered_data.loc[:, 'MKTSENTIMENTS'].apply(label_check)

# 负采样
'''
欠采样强行让正负情感标签数相等
'''
posi_data = filtered_data[filtered_data['MKTSENTIMENTS'].isin([1])]
nege_data = filtered_data[filtered_data['MKTSENTIMENTS'].isin([0])]
if posi_data.shape[0] >= nege_data.shape[0]:
    posi_data = posi_data.sample(frac=1).iloc[:nege_data.shape[0], :] # 数据随机后取前n条
else:
    nege_data = nege_data.sample(frac=1).iloc[:posi_data.shape[0], :]

filtered_data = pd.concat([posi_data, nege_data], ignore_index=True).sample(frac=1)

filtered_data.to_csv('../temp_data/WIND_FinancialNews_cleaned/processed_data.csv', index=False) # 得到处理后数据

'''
输出正负新闻比例，以及新闻的最大文本长度
'''
# filtered_data = pd.read_csv('../data/WIND_FinancialNews_cleaned/processed_data.csv')
num_data = filtered_data.values.shape[0]
num_posi = np.sum(filtered_data.loc[:, 'MKTSENTIMENTS'])
num_nege = num_data - num_posi
print("可用新闻%d条" % num_data)
print("其中，正负面新闻分别占%.2f和%.2f" % (num_posi / num_data, 1 - num_posi / num_data))

# 计算最长文本
calcu_length = lambda x: len(x)
text_length = filtered_data.loc[:, 'CONTENT'].apply(calcu_length)
print("最大文本长度为%d" % max(text_length))