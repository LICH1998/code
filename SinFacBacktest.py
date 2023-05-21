# -*- coding: utf-8 -*-
# @author: ChenWei
# @date: 2021/07/30
# @filename: SinFacBacktest.py
# @software: Pycharm

"""简单单因子分层回测"""

import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import matplotlib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.rc("font", family='Kaiti')
matplotlib.rcParams['axes.unicode_minus'] = False



class General_Data():
    """存储通用回测行情数据"""

    def __init__(self, param):
        self.param = param

        # 读取数据
        self.all_stock_code = pd.read_csv(param['path_stock_list'], index_col=0)['0'].tolist()
        self.stock_pool = pd.read_csv(param['path_stock_pool'], index_col=0).reindex(self.all_stock_code)
        self.close_adj = pd.read_csv(param['path_dailyinfo_close_adj'], index_col=0).reindex(self.all_stock_code)
        self.turn = pd.read_csv(param['path_dailyinfo_turn'], index_col=0).reindex(self.all_stock_code)
        self.delist_date = pd.read_csv(param['path_delist_date'], index_col=0)['delist_date'].reindex(self.all_stock_code)
        self.dailyinfo_dates = pd.read_csv(param['path_dailyinfo_dates'], index_col=0)['0']


class LayerBacktest():

    def __init__(self, gn_data):
        """
        初始化函数
        """
        self.param = gn_data.param
        self.fee = 0.002  #手续费为千分之2

        # Part1：读取通用行情数据
        self.all_stock_code = gn_data.all_stock_code
        self.stock_pool = gn_data.stock_pool
        self.close_adj = gn_data.close_adj
        self.turn = gn_data.turn
        self.delist_date = gn_data.delist_date
        self.dailyinfo_dates = gn_data.dailyinfo_dates

        # Part2：局部变量
        self.df_factor = None  # 因子数据
        self.layer_num = None  # 分层层数
        self.suspend_correct = None  # 截面期是否进行权重修正

        # Part3：回测变量
        self.stock_each_layer = None  # 每层股票数
        self.weight_each_layer_stock = None  # 每层单只个股权重

        # Part4：结果变量
        self.df_stock_layer = None
        self.dates_trade = None
        self.df_layer_backtest = None

    def ps_load_df_factor(self, factor):
        """读取因子数据矩阵"""
        self.df_factor = factor.reindex(self.all_stock_code)

    def ps_load_bkt_status(self, layer_num=3, suspend_correct=True):
        """
        设置分层回测的参数
        layer_num: 分层的数量
        suspend_correct: 是否在截面期进行权重修正
        """
        self.layer_num = layer_num
        self.suspend_correct = suspend_correct

    def ps_filter_date(self):
        """时间截取"""
        assert self.df_factor is not None, '请先读取因子数据！'
        self.df_factor = self.df_factor.loc[:, self.param['start_date']: self.param['end_date']]
        self.stock_pool = self.stock_pool.loc[:, self.param['start_date']: self.param['end_date']]

    def ps_generate_stock_layers(self):
        """计算各层的持仓权重"""
        # 计算月末权重
        df_stock_layer = dict()
        for k, date in enumerate(self.df_factor.columns):
            se_factor = self.df_factor.loc[:, date].reindex(self.stock_pool.loc[:, date][(self.stock_pool.loc[:, date].abs() > 0)].index).dropna()
            self.stock_each_layer = int(len(se_factor) / self.layer_num)
            self.weight_each_layer_stock = 1/self.stock_each_layer if self.stock_each_layer >= 10 else 0.1  # 确保单只股票权重不超过10%

            # 对每一层进行循环
            for i in range(self.layer_num):
                ls_stock = se_factor.sort_values(ascending=False).iloc[self.stock_each_layer * i: self.stock_each_layer * (i+1)].index.values

                if k == 0:
                    df_stock_layer[i] = pd.DataFrame({date: self.weight_each_layer_stock}, index=ls_stock).reindex(self.df_factor.index)
                else:
                    df_stock_layer[i][date] = pd.Series(self.weight_each_layer_stock, index=ls_stock).reindex(self.df_factor.index)

        # 映射到月初权重
        for i in range(self.layer_num):
            idx_month_end = [list(self.dailyinfo_dates).index(x) for x in df_stock_layer[i].columns]
            try:
                df_stock_layer[i].columns = [self.dailyinfo_dates[x+1] for x in idx_month_end]
            except KeyError:
                df_stock_layer[i].columns = [self.dailyinfo_dates[x+1] for x in idx_month_end[:-1]] + [df_stock_layer[i].columns[-1]]

        self.df_stock_layer = df_stock_layer
        self.dates_trade = self.df_stock_layer[0].columns

    def ps_layer_backtest(self):
        """分层回测"""
        self.df_layer_backtest = pd.DataFrame()

        for i in range(self.layer_num):
            # 回测起止日期的序号
            id_date_bkt_start = np.where(self.dailyinfo_dates == self.df_factor.columns[0])[0][0]+1
            id_date_bkt_end = np.where(self.dailyinfo_dates == self.dailyinfo_dates[self.dailyinfo_dates <= self.param['end_date']].iloc[-1])[0][0]

            # 换仓日的序号
            id_dates_trade = [np.where(self.dailyinfo_dates == k)[0][0] for k in self.dates_trade]

            # 储存净值
            value_daily_raw = pd.Series(np.nan, index=self.dailyinfo_dates)

            # 第一个换仓日及上个交易日的净值置为1
            value_daily_raw.iloc[id_date_bkt_start - 1:id_date_bkt_start + 1] = 1

            # 储存上期权重，用于计算换手率；初始值置为0
            weight_last = pd.Series(0, index=self.df_factor.index)

            # 储存换手率
            turnover = pd.Series(0, index=self.dates_trade)

            # 实际权重
            real_weight = pd.DataFrame(0, index=self.df_stock_layer[i].index, columns=self.df_stock_layer[i].columns)

            # 正式回测
            # 遍历换仓日
            for i_date_trade, id_date_trade in enumerate(id_dates_trade):

                # ------------------------------------------#
                # 1. 确定每次换仓日对应的净值日区间
                # 换仓日的序号
                if id_date_trade < id_dates_trade[-1]:
                    # 若非最后一个换仓日，则净值日为换仓日至下一个换仓日
                    id_dates_value = list(range(id_date_trade, id_dates_trade[i_date_trade + 1] + 1))
                elif id_date_trade < id_date_bkt_end:
                    # 若为最后一个换仓日，则净值日为换仓日至回测终止日
                    id_dates_value = list(range(id_date_trade, id_date_bkt_end + 1))
                else:
                    id_dates_value = []
                # ------------------------------------------#

                # 2. 针对停牌、退市等特殊情况进行权重修正
                if not self.suspend_correct:
                    # 不进行修正
                    real_weight.iloc[:, i_date_trade] = self.df_stock_layer[i].iloc[:, i_date_trade].fillna(0).values
                else:
                    # 进行修正
                    # 是否不可交易，1为不可交易
                    flag_non_trade = ~(self.turn.iloc[:, id_date_trade] > 1e-4)
                    # 是否退市，1为退市
                    flag_delist = (self.delist_date <= self.dailyinfo_dates[id_date_trade])
                    # 当期理论权重
                    weight_now = self.df_stock_layer[i].iloc[:, i_date_trade].fillna(0)
                    # 当期理论总权重
                    weight_now_target = sum(weight_now)
                    # 上一期末组合权重
                    # 买卖股票是否可交易
                    weight_trade = weight_now - weight_last
                    if weight_trade.loc[flag_non_trade].sum() == 0:
                        # 若买卖均可交易，则实际买入权重为理论买入权重
                        real_weight.iloc[:, i_date_trade] = weight_now
                    else:
                        # 若存在不可交易的股票
                        # print('第%d期（%s）存在不可交易股票' % (i_date_trade + 1, self.dates_trade[i_date_trade]))
                        # ------------------------------------------ #
                        # Step1 上一期有持仓，但是当期调仓日无法交易的，仓位保持不变
                        id_stock_non_trade1 = np.all([weight_last != 0, flag_non_trade == True, flag_delist == False], axis=0)
                        id_stock_non_trade1 = np.where(id_stock_non_trade1 == True)[0]
                        real_weight.iloc[id_stock_non_trade1, i_date_trade] = weight_last.iloc[id_stock_non_trade1]
                        weight_now.iloc[id_stock_non_trade1] = 0
                        # 若存在上期有持仓且已退市的股票，假设通过某种途径已平仓
                        id_stock_delist = np.all([weight_last != 0, flag_non_trade == True, flag_delist == True], axis=0)
                        # 输出退市股票
                        if sum(id_stock_delist) > 0:
                            id_stock_delist = np.where(id_stock_delist == True)[0]
                            for i_stock in id_stock_delist:
                                print('%s退市卖出' % (self.all_stock_code[i_stock]))

                        # ------------------------------------------ #
                        # Step2：上一期没有持仓，当期新增持仓但无法交易的，新增持仓置为0
                        id_stock_non_trade2 = np.all([weight_now != 0, flag_non_trade == True], axis=0)
                        weight_now.loc[id_stock_non_trade2] = 0

                        # 输出新增持仓无法交易的股票
                        if sum(id_stock_non_trade2) > 0:
                            id_stock_non_buy = np.where(id_stock_non_trade2 == True)[0]
                            # for i_stock in id_stock_non_buy:
                            #     print('%s新增持仓无法交易' % (self.all_stock_code[i_stock]))

                        # ------------------------------------------ #
                        # Step3：实际当期真正权重 = 理论当期总权重 - 上一期无法交易的权重
                        weight_now_real = weight_now_target - sum(real_weight.iloc[:, i_date_trade])
                        # ------------------------------------------ #
                        # Step4：修正当期权重，使得总仓位与目标仓位一致
                        if sum(weight_now) != 0:
                            weight_now = weight_now * weight_now_real / sum(weight_now)

                        # ------------------------------------------ #
                        # Step5：买入
                        id_stock_now = np.all([weight_now != 0, flag_non_trade == False], axis=0)
                        id_stock_now = np.where(id_stock_now == True)[0]
                        real_weight.iloc[id_stock_now, i_date_trade] = weight_now.iloc[id_stock_now]

                # ------------------------------------------#
                # 3. 记录权重、买入价
                weight_now = real_weight.iloc[:, i_date_trade]
                price_buy = self.close_adj.iloc[:, id_date_trade]

                # 用于计算净值的基准value = 当期换仓日value
                # 当期换仓日value由上期换仓日的最后一个净值日根据卖出价计算得到
                # 当期换仓日value在随后计算时会根据收盘价更新
                value_port = value_daily_raw.iloc[id_date_trade]

                turnover.iloc[i_date_trade] = sum(abs(weight_now.values - weight_last.values))

                # 根据换手率扣除交易费用
                value_port = value_port * (1 - turnover.iloc[i_date_trade] * self.fee)
                # ------------------------------------------#
                # 4. 计算每日净值
                for id_date_value in id_dates_value:
                    if id_date_value < id_dates_value[-1]:
                        # 非当期最后一个净值日
                        price_value = self.close_adj.iloc[:, id_date_value]
                    else:
                        price_value = self.close_adj.iloc[:, id_date_value]

                        # 计算自然增长的权重，用于计算换手率和交易费用
                        weight_last = weight_now * price_value / price_buy
                        weight_last.loc[np.isnan(weight_last)] = 0
                        # 权重归一化，考虑现金仓位
                        weight_last = weight_last / (sum(weight_last) + 1 - sum(weight_now))  # 这样归一化可以把杠杆也统一容纳进来
                        # print('sum of weight last:{}'.format(weight_last.sum()))

                    # 计算收益
                    returns_port = np.nansum(weight_now * (price_value / price_buy - 1))

                    # 计算净值；若为当期换仓日的最后一个净值日，则此净值为下期换仓日计算净值的基准
                    value_daily_raw.iloc[id_date_value] = value_port * (1 + returns_port)

            print('分层{}回测完成'.format(i+1))
            self.df_layer_backtest['分层{}'.format(i+1)] = value_daily_raw.dropna()

    def go(self, plot=False, disp=False):
        self.ps_filter_date()
        self.ps_generate_stock_layers()
        self.ps_layer_backtest()
        self.df_layer_backtest.to_excel(self.param['path_result'])

        # 输出
        if disp:
            print(self.df_layer_backtest)

        # 绘图
        if plot:
            self.df_layer_backtest.plot(figsize = (12, 4))
            plt.show()
            # plt.savefig(self.param['path_figure'])

        return self.df_layer_backtest


if __name__ == '__main__':
    # 分层回测
    param = dict()
    param['path_general'] = './info/'
    param['path_result'] = 'result_real.xlsx'
    param['start_date'] = '2012-02-28'
    param['end_date'] = '2023-02-28'
    param['path_stock_pool'] = 'Average_jor_factor.csv'

    param['path_stock_list'] = param['path_general'] + 'stock_code_info.csv'  #股票列表
    param['path_dailyinfo_dates'] = param['path_general'] + 'day.csv'           #交易日列表
    param['path_dailyinfo_close_adj'] = param['path_general'] + 'close_adj_day.csv'   #复权收盘价
    param['path_dailyinfo_turn'] = param['path_general'] + 'turn_day.csv'       #换手率
    param['path_delist_date'] = param['path_general'] + 'delist_date_info.csv'    #退市信息

    # 单因子分层回测
    gn_data = General_Data(param)
    lbt = LayerBacktest(gn_data)

    # 这一部分就可以更便捷地对多个因子进行分层回测
    # factor = pd.read_csv('../result/factor/【华泰金工】clean_sue_txt.csv', index_col=0)
    factor = pd.read_csv('Average_jor_factor.csv', index_col=0)

    lbt.ps_load_df_factor(factor)
    lbt.ps_load_bkt_status(layer_num=10, suspend_correct=True)
    df_layer_bkt = lbt.go(plot=True, disp=True)

    # 业绩
    from performance import Metrics
    perf = pd.DataFrame(columns=['绝对收益', '超额收益', '最大回撤', '夏普比率', '卡玛比率'])
    for col in df_layer_bkt:
        perf.loc[col, '绝对收益'] = Metrics.annual_return(df_layer_bkt[col])

    # 计算相对于中证500的超额净值
    zz500 = pd.read_csv('./Data/zz500.csv', index_col=0).iloc[:, 0]
    zz500.index = [datetime.strptime(i, "%Y/%m/%d").strftime("%Y-%m-%d") for i in zz500.index]
    df_layer_bkt['base'] = zz500.loc[(zz500.index <= param['end_date']) & (zz500.index > param['start_date'])]
    df_layer_bkt['base'] = df_layer_bkt['base'] / df_layer_bkt['base'].iloc[0]
    for col in df_layer_bkt.columns[:-1]:
        df_layer_bkt[col] = df_layer_bkt[col] / df_layer_bkt['base']

    df_layer_bkt = df_layer_bkt.iloc[:, :]
    df_layer_bkt.to_excel('backtest_excess.xlsx')
    # df_layer_bkt.to_excel('../result/backtest_by_layers_clean_sue_txt_excess.xlsx')

    # 业绩
    for col in df_layer_bkt:
        perf.loc[col, '超额收益'] = Metrics.annual_return(df_layer_bkt[col])
        perf.loc[col, '最大回撤'] = Metrics.max_drawdown(df_layer_bkt[col])
        perf.loc[col, '夏普比率'] = Metrics.sharpe_ratio(df_layer_bkt[col])
        perf.loc[col, '卡玛比率'] = Metrics.calmar_ratio(df_layer_bkt[col])

    perf.to_excel('performance.xlsx')

    print(perf)




