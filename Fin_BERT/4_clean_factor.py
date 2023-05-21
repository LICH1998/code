# -*- coding: utf-8 -*-
"""
@author: HTSC
"""

import warnings
import numpy as np
import pandas as pd


warnings.filterwarnings('ignore')


def cal_clean_factor(start_date, end_date):
    """
    将上一步预测出的raw_sue_txt因子值转换为按月和个股存储的clean_sue_txt因子文件
    :param start_date: 开始日期，eg.2012-01-31
    :param end_date: 结束日期, eg.2022-04-29
    :example:
    >>> start_date = '2012-01-31'
    >>> end_date = '2022-04-29'
    """
    # 回看几个月
    lookback = 3

    # 读取数据
    stock_list = pd.read_csv('../raw_data/general_data/stock_code_info.csv', index_col=0)['0'].tolist()
    monthly_dates = pd.read_csv('../data/month.csv', index_col=0)['0'].tolist()

    raw_factor = pd.read_csv('../result/raw_factor.csv', index_col=0)
    raw_factor['STOCK_CODE'] = raw_factor['STOCK_CODE'].astype(str).str.rjust(6, '0')

    # 对截面期进行循环
    stock_list_no_suffix = [x[:6] for x in stock_list]
    ls_month = list(filter(lambda x: (x >= start_date) & (x <= end_date), monthly_dates))
    clean_factor = pd.DataFrame(columns=ls_month, index=stock_list_no_suffix)

    for i_month in ls_month:
        curr_date_start = monthly_dates[monthly_dates.index(i_month) - lookback]
        curr_date_end = i_month
        tmp_factor = raw_factor[(raw_factor['REPORT_DATE'] > curr_date_start) & (raw_factor['REPORT_DATE'] <= curr_date_end)].groupby('STOCK_CODE').agg({'factor': np.mean})
        clean_factor.loc[tmp_factor.index, i_month] = tmp_factor.iloc[:, 0]

    clean_factor.index = stock_list
    clean_factor.to_csv('../result/clean_factor.csv')

    return clean_factor


if __name__ == '__main__':
    start_date = '2009-01-23'
    end_date = '2022-06-30'
    clean_factor = cal_clean_factor(start_date, end_date)
    
    for col in clean_factor:
        if '2015-11-30' <= col <= '2016-09-30':
            clean_factor.loc[:, col] = clean_factor['2015-10-30']
    clean_factor.to_csv('../result/clean_forecast_adjust_txt.csv')