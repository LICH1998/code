# -*- coding: utf-8 -*-
"""
@author: HTSC
"""

import os
import datetime
import pandas as pd
import numpy as np
from dateutil.parser import parse


def preprocess_data_1(param):
    """
    数据预处理，拼接分析师预期调整样本和对应的研报样本
    """
    ls_file = os.listdir(param['path_cmb_report_adjust'])
    for file in ls_file:
        date = file[-13:-5]

        # 更新模式下文件已经处理过则跳过
        if param['update'] and os.path.exists(param['path_cmb_report_adjust_adj'] + 'CMB_REPORT_ADJUST_adj_{}.xlsx'.format(date)):
            print('{} already precoess'.format(date))
            continue
        else:
            # 读取原数据
            report_adjust = pd.read_excel(param['path_cmb_report_adjust'] + file, dtype={'STOCK_CODE': str}, index_col=0)
            report_research = pd.read_excel(param['path_cmb_report_research'] + 'GOGOAL_CMB_REPORT_RESEARCH_{}.xlsx'.format(date), dtype={'CODE': str})
            if len(report_adjust) == 0 or len(report_research) == 0:
                report_adjust_adj = pd.DataFrame()
                report_adjust_adj.to_excel(param['path_cmb_report_adjust_adj'] + 'CMB_REPORT_ADJUST_adj_{}.xlsx'.format(date))
                print(date + ' preprocess finish!')
                continue

            # 数据格式调整
            report_adjust = report_adjust.drop_duplicates(subset=['REPORT_ID', 'STOCK_CODE'], keep='first').reset_index(drop=True)
            report_research = report_research.rename(columns={'CODE': 'STOCK_CODE', 'ID': 'REPORT_ID'})
            report_research = report_research[['REPORT_ID', 'TITLE', 'CONTENT', 'ATTENTION_NAME']] # ATTENTION_NAME: 一般报告, 首次关注, 首份报告

            # 根据REPORT_ID合并两组数据
            report_adjust_adj = pd.merge(report_adjust, report_research, on='REPORT_ID', how='inner').reset_index(drop=True)
        
            # 删除部分样本，删除逻辑：
            # 1.删除首盖，首盖报告并不一定会随着突发事件进行，增加噪音；
            # 2.删除盈利预测不变的样本
            # 3.删除港股
            report_adjust_adj = report_adjust_adj[report_adjust_adj['ATTENTION_NAME'].apply(lambda x: '首' not in x)]
            if len(report_adjust_adj) == 0:
                report_adjust_adj = pd.DataFrame()
                report_adjust_adj.to_excel(param['path_cmb_report_adjust_adj'] + 'CMB_REPORT_ADJUST_adj_{}.xlsx'.format(date))
                print(date + ' preprocess finish!')
                continue
            report_adjust_adj = report_adjust_adj[(report_adjust_adj['CURRENT_FORECAST_PROFIT'] != report_adjust_adj['PREVIOUS_FORECAST_PROFIT']) &
                                                  (~pd.isnull(report_adjust_adj['PREVIOUS_FORECAST_PROFIT']))]
            if len(report_adjust_adj) == 0:
                report_adjust_adj = pd.DataFrame()
                report_adjust_adj.to_excel(param['path_cmb_report_adjust_adj'] + 'CMB_REPORT_ADJUST_adj_{}.xlsx'.format(date))
                print(date + ' preprocess finish!')
                continue
            report_adjust_adj = report_adjust_adj[report_adjust_adj['STOCK_CODE'].apply(lambda x: len(x) == 6)] # 6位编码, 删除港股

            # 保存结果
            report_adjust_adj = report_adjust_adj.reset_index(drop=True)
            report_adjust_adj.to_excel(param['path_cmb_report_adjust_adj'] + 'CMB_REPORT_ADJUST_adj_{}.xlsx'.format(date))

            print(date + ' preprocess finish!')


def preprocess_data_2(param):
    """
    数据预处理，拼接分析师评级调整样本和对应的研报样本
    """
    ls_file = os.listdir(param['path_cmb_report_score_adjust'])
    for file in ls_file:
        date = file[-13:-5]

        # 更新模式下文件已经处理过则跳过
        if param['update'] and os.path.exists(param['path_cmb_report_score_adjust_adj'] + 'CMB_REPORT_SCORE_ADJUST_adj_{}.xlsx'.format(date)):
            print('{} already precoess'.format(date))
            continue
        else:
            # 读取原数据
            report_score_adjust = pd.read_excel(param['path_cmb_report_score_adjust'] + file, dtype={'STOCK_CODE': str}, index_col=0)

            try:
                report_research = pd.read_excel(param['path_cmb_report_research'] + 'GOGOAL_CMB_REPORT_RESEARCH_{}.xlsx'.format(date), dtype={'CODE': str})
            except FileNotFoundError:
                continue

            if len(report_score_adjust) == 0 or len(report_research) == 0:
                report_score_adjust_adj = pd.DataFrame()
                report_score_adjust_adj.to_excel(param['path_cmb_report_score_adjust_adj'] + 'CMB_REPORT_SCORE_ADJUST_adj_{}.xlsx'.format(date))
                print(date + ' preprocess finish!')
                continue

            # 数据格式调整
            report_score_adjust = report_score_adjust.drop_duplicates(subset=['REPORT_ID', 'STOCK_CODE'], keep='first').reset_index(drop=True)
            report_research = report_research.rename(columns={'CODE': 'STOCK_CODE', 'ID': 'REPORT_ID'})
            report_research = report_research[['REPORT_ID', 'TITLE', 'CONTENT', 'ATTENTION_NAME']]

            # 根据REPORT_ID合并两组数据
            report_score_adjust_adj = pd.merge(report_score_adjust, report_research, on='REPORT_ID', how='inner').reset_index(drop=True)
            if len(report_score_adjust_adj) == 0:
                report_score_adjust_adj = pd.DataFrame()
                report_score_adjust_adj.to_excel(param['path_cmb_report_score_adjust_adj'] + 'CMB_REPORT_SCORE_ADJUST_adj_{}.xlsx'.format(date))
                print(date + ' preprocess finish!')
                continue

            # 删除部分样本，删除逻辑：
            # 1.删除首盖，首盖报告并不一定会随着突发事件进行，增加噪音；
            # 2.删除港股
            report_score_adjust_adj = report_score_adjust_adj[report_score_adjust_adj['ATTENTION_NAME'].apply(lambda x: '首' not in x)]
            report_score_adjust_adj = report_score_adjust_adj[report_score_adjust_adj['STOCK_CODE'].apply(lambda x: len(x) == 6)]

            # 保存结果
            report_score_adjust_adj = report_score_adjust_adj.reset_index(drop=True)
            report_score_adjust_adj.to_excel(param['path_cmb_report_score_adjust_adj'] + 'CMB_REPORT_SCORE_ADJUST_adj_{}.xlsx'.format(date))

            print(date + ' preprocess finish!')


def generate_feature_1(param):
    """
    将标题与内容合并，截取需要的列
    """
    ls_file = os.listdir(param['path_cmb_report_adjust_adj'])
    for file in ls_file:
        # 计算日期
        date = file[-13:-5]

        # 判断是否已经进行过分词处理
        if os.path.exists(param['path_cmb_report_adjust_adj_split_word'] + 'CMB_REPORT_ADJUST_adj_SPLIT_WORD_{}.xlsx'.format(parse(date).strftime('%Y%m%d'))):
            continue

        # 读取每日研报文件
        rpt_research = pd.read_excel(param['path_cmb_report_adjust_adj'] + file, index_col=0)

        # 如果数据大小不为空
        if rpt_research.shape[0] != 0:
            # 截取研报原始数据需要的列并处理格式
            rpt_research = rpt_research[['STOCK_CODE', 'TITLE', 'CONTENT']]
            rpt_research.insert(0, 'REPORT_DATE', date)
            rpt_research['REPORT_DATE'] = pd.to_datetime(rpt_research['REPORT_DATE'])
            rpt_research['STOCK_CODE'] = rpt_research['STOCK_CODE'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))

            # 将标题与内容合并
            rpt_research.loc[:, 'CONTENT'] = rpt_research.loc[:, 'TITLE'] + '。' + rpt_research.loc[:, 'CONTENT']
            del rpt_research['TITLE']

            # 整理格式，截取需要的列
            rpt_research = rpt_research.reset_index(drop=True)
            rpt_research = rpt_research[['REPORT_DATE', 'STOCK_CODE', 'CONTENT']]

            # 输出数据
            rpt_research.to_excel(param['path_cmb_report_adjust_adj_split_word'] + 'CMB_REPORT_ADJUST_adj_SPLIT_WORD_{}.xlsx'.format(parse(date).strftime('%Y%m%d')), encoding='utf_8_sig')

        else:
            rpt_research.to_excel(param['path_cmb_report_adjust_adj_split_word'] + 'CMB_REPORT_ADJUST_adj_SPLIT_WORD_{}.xlsx'.format(parse(date).strftime('%Y%m%d')), encoding='utf_8_sig')
        print(date + ' feature finish!')


def generate_feature_2(param):
    """
    将标题与内容合并，截取需要的列
    """
    ls_file = os.listdir(param['path_cmb_report_score_adjust_adj'])
    for file in ls_file:
        # 计算日期
        date = file[-13:-5]

        # 判断是否已经进行过分词处理
        if os.path.exists(param['path_cmb_report_score_adjust_adj_split_word'] + 'CMB_REPORT_SCORE_ADJUST_adj_SPLIT_WORD_{}.xlsx'.format(parse(date).strftime('%Y%m%d'))):
            continue

        # 读取每日研报文件
        rpt_research = pd.read_excel(param['path_cmb_report_score_adjust_adj'] + file, index_col=0)

        # 如果数据大小不为空
        if rpt_research.shape[0] != 0:
            # 截取研报原始数据需要的列并处理格式
            rpt_research = rpt_research[['STOCK_CODE', 'TITLE', 'CONTENT']]
            rpt_research.insert(0, 'REPORT_DATE', date)
            rpt_research['REPORT_DATE'] = pd.to_datetime(rpt_research['REPORT_DATE'])
            rpt_research['STOCK_CODE'] = rpt_research['STOCK_CODE'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))

            # 将标题与内容合并
            rpt_research.loc[:, 'CONTENT'] = rpt_research.loc[:, 'TITLE'] + '。' + rpt_research.loc[:, 'CONTENT']
            del rpt_research['TITLE']

            # 整理格式，截取需要的列
            rpt_research = rpt_research.reset_index(drop=True)
            rpt_research = rpt_research[['REPORT_DATE', 'STOCK_CODE', 'CONTENT']]

            # 输出数据
            rpt_research.to_excel(param['path_cmb_report_score_adjust_adj_split_word'] + 'CMB_REPORT_SCORE_ADJUST_adj_SPLIT_WORD_{}.xlsx'.format(parse(date).strftime('%Y%m%d')), encoding='utf_8_sig')

        else:
            rpt_research.to_excel(param['path_cmb_report_score_adjust_adj_split_word'] + 'CMB_REPORT_SCORE_ADJUST_adj_SPLIT_WORD_{}.xlsx'.format(parse(date).strftime('%Y%m%d')), encoding='utf_8_sig')
        print(date + 'feature finish!')


def generate_target(param, df):
    """
    计算超额收益,在后续函数中调用
    """
    # 计算每条样本AR计算区间
    dailyinfo_dates = pd.read_csv(param['dailyinfo_dates'], index_col=0)['0']
    dailyinfo_dates.index = dailyinfo_dates
    dailyinfo_all_dates = pd.date_range(dailyinfo_dates.iloc[0], dailyinfo_dates.iloc[-1])
    dailyinfo_dates = dailyinfo_dates.reindex(dailyinfo_all_dates.strftime('%Y-%m-%d'))

    # 匹配前一个和后一个交易日
    df['lst_trade_dt'] = dailyinfo_dates.shift(1).fillna(method='ffill').reindex(df['REPORT_DATE'].astype(str)).values
    df['nxt_trade_dt'] = dailyinfo_dates.shift(-1).fillna(method='bfill').reindex(df['REPORT_DATE'].astype(str)).values

    # 计算AR
    dailyinfo_close_adj = pd.read_csv(param['dailyinfo_close'], index_col=0)
    dailyinfo_close_adj.index = [x[:6] for x in dailyinfo_close_adj.index]
    benchmark = pd.read_csv(param['benchmark']).T.iloc[1:,0]

    df['stock_ret'] = df.apply(lambda x: dailyinfo_close_adj.loc[x['STOCK_CODE'], x['nxt_trade_dt']] / dailyinfo_close_adj.loc[x['STOCK_CODE'], x['lst_trade_dt']] - 1
                               if (x['STOCK_CODE'] in dailyinfo_close_adj.index) and (not pd.isnull(x['nxt_trade_dt'])) and (not pd.isnull(x['lst_trade_dt'])) else np.nan, axis=1)
    df['bench_ret'] = df.apply(lambda x: benchmark[x['nxt_trade_dt']] / benchmark[x['lst_trade_dt']] - 1
                               if (not pd.isnull(x['nxt_trade_dt'])) and (not pd.isnull(x['lst_trade_dt'])) else np.nan, axis=1)

    df['AR'] = df['stock_ret'] - df['bench_ret']
    df = df.dropna(subset=['AR', 'stock_ret', 'bench_ret'])

    return df


def generate_sample_1(param):
    """
    生成分析师预期调整的样本
    """
    ls_file = os.listdir(param['path_cmb_report_adjust_adj_split_word'])

    # 初始模式
    if param['update'] == 0:
        df = pd.DataFrame()

    # 更新模式
    else:
        df = pd.read_csv(param['path_cmb_report_adjust_adj_split_word_allinfo'])[['REPORT_DATE', 'STOCK_CODE', 'CONTENT']]
        param['date_start'] = df['REPORT_DATE'].sort_values(ascending=True).values[-1]
        param['date_start'] = (parse(param['date_start']) + datetime.timedelta(days=-3)).strftime('%Y%m%d')
        df = df[df['REPORT_DATE'] <= parse(param['date_start']).strftime('%Y-%m-%d')]
        df['STOCK_CODE'] = df['STOCK_CODE'].astype(str).str.rjust(6, '0')

    for file in ls_file:
        date = file[-13:-5]
        if date > param['date_start']:
            df_new = pd.read_excel(param['path_cmb_report_adjust_adj_split_word'] + file, index_col=0)
            if len(df_new) == 0:
                continue

            df_new = df_new[['REPORT_DATE', 'STOCK_CODE', 'CONTENT']]
            df = pd.concat([df, df_new], axis=0)
            print(date + ' sample finish!')
            df = df.reset_index(drop=True)

    # 计算标签Y
    df['STOCK_CODE'] = df['STOCK_CODE'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])
    df = generate_target(param, df)
    print('匹配超额收益完成')

    # 输出结果
    df = df.dropna().reset_index(drop=True)
    df.to_csv(param['path_cmb_report_adjust_adj_split_word_allinfo'], encoding='utf_8_sig')

def generate_sample_2(param):
    """
    生成分析师预期调整的样本
    """
    ls_file = os.listdir(param['path_cmb_report_score_adjust_adj_split_word'])

    # 初始模式
    if param['update'] == 0:
        df = pd.DataFrame()

    # 更新模式
    else:
        df = pd.read_csv(param['path_cmb_report_score_adjust_adj_split_word_allinfo'])[['REPORT_DATE', 'STOCK_CODE', 'CONTENT']]
        param['date_start'] = df['REPORT_DATE'].sort_values(ascending=True).values[-1]
        param['date_start'] = (parse(param['date_start']) + datetime.timedelta(days=-3)).strftime('%Y%m%d')
        df = df[df['REPORT_DATE'] <= parse(param['date_start']).strftime('%Y-%m-%d')]
        df['STOCK_CODE'] = df['STOCK_CODE'].astype(str).str.rjust(6, '0')

    for file in ls_file:
        date = file[-13:-5]
        if date > param['date_start']:
            df_new = pd.read_excel(param['path_cmb_report_score_adjust_adj_split_word'] + file, index_col=0)
            if len(df_new) == 0:
                continue

            df_new = df_new[['REPORT_DATE', 'STOCK_CODE', 'CONTENT']]
            df = pd.concat([df, df_new], axis=0)
            print(date + ' sample finish!')
            df = df.reset_index(drop=True)

    # 计算标签Y
    df['STOCK_CODE'] = df['STOCK_CODE'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])
    df = generate_target(param, df)
    print('匹配超额收益完成')

    # 输出结果
    df = df.dropna().reset_index(drop=True)
    df.to_csv(param['path_cmb_report_score_adjust_adj_split_word_allinfo'], encoding='utf_8_sig')


if __name__ == '__main__':
    
    # 设置参数
    param = dict()
    param['date_start'] = '20061231'  # 初次固定，后续无需修改
    param['update'] = 0  # 初始模式or更新模式， 1——更新模式，0——初始模式
    param['dailyinfo_dates'] = '../raw_data/general_data/day.csv'
    param['dailyinfo_close'] = '../raw_data/general_data/close_adj_day.csv'
    param['benchmark'] = '../raw_data/general_data/csi500_day.csv'

    # 分析师盈利预测调整样本
    param['path_cmb_report_adjust'] = '../raw_data/GOGOAL/CMB_REPORT_ADJUST/'
    param['path_cmb_report_research'] = '../raw_data/GOGOAL/CMB_REPORT_RESEARCH/'
    param['path_cmb_report_adjust_adj'] = '../temp_data/CMB_REPORT_ADJUST_adj/'
    param['path_cmb_report_adjust_adj_split_word'] = '../temp_data/CMB_REPORT_ADJUST_adj_SPLIT_WORD/'
    param['path_cmb_report_adjust_adj_split_word_allinfo'] = '../temp_data/data_report_adjust_split_word_allinfo_origin.csv'

    # 分析评级调整样本
    param['path_cmb_report_score_adjust'] = '../raw_data/GOGOAL/CMB_REPORT_SCORE_ADJUST/'
    param['path_cmb_report_score_adjust_adj'] = '../temp_data/CMB_REPORT_SCORE_ADJUST_adj/'
    param['path_cmb_report_score_adjust_adj_split_word'] = '../temp_data/CMB_REPORT_SCORE_ADJUST_adj_SPLIT_WORD/'
    param['path_cmb_report_score_adjust_adj_split_word_allinfo'] = '../temp_data/data_report_score_adjust_split_word_allinfo_origin.csv'
        
    # 执行数据清洗
    preprocess_data_1(param)
    generate_feature_1(param)
    generate_sample_1(param)

    # preprocess_data_2(param)
    # generate_feature_2(param)
    # generate_sample_2(param)