#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-03-07 22:39
# @Author : lichangheng

"""回测代码"""

import pandas as pd
import numpy as np


def init_param():
    """
    初始化参数
    >>> param = init_param()
    """
    # 回测参数 ----------
    param = dict()
    param['bkt_start'] = '2013-01-04'  # 回测开始时间
    param['bkt_end'] = '2022-05-31'  # 回测结束时间
    param['fee'] = 0.0015  # 单边千一点五
    param['suspend_correct'] = 1  # 是否进行停牌修正
    param['type_price'] = 'vwap'  # 'close' 'vwap'

    # 路径参数 ----------
    path = 'D:/HTSC/Dataset/result/'
    param['stock_code'] = path + 'basicinfo_stock_number_wind.csv'  # 股票列表
    param['dailyinfo_dates'] = path + 'dailyinfo_dates.csv'  # 日频日期序列
    param['dailyinfo_close'] = path + 'dailyinfo_close.csv'  # 日频收盘价
    param['dailyinfo_close_adj'] = path + 'dailyinfo_close_adj.csv'  # 日频复权收盘价
    param['dailyinfo_vwap'] = path + 'dailyinfo_1_vwap.csv'  # 日频收盘价
    param['dailyinfo_turn'] = path + 'dailyinfo_turn.csv'  # 日频换手率
    param['basicinfo_delist_date'] = path + 'basicinfo_delist_date.csv'  # 退市时间

    # 权重路径
    param['optimal_weight'] = ''
    param['opt_bkt'] = 'optimal_backtest.xlsx'

    return param


def BackTest(param):

    # 读取数据
    stock_code = pd.read_csv(param['stock_code'], index_col=0).iloc[:, 0]
    daily_dates = pd.read_csv(param['dailyinfo_dates'], index_col=0).iloc[:, 0].to_list()
    daily_dates = pd.to_datetime(daily_dates)

    close_adj = pd.read_csv(param['dailyinfo_close_adj'], index_col=0)
    turn = pd.read_csv(param['dailyinfo_turn'], index_col=0)

    if param['type_price'] == 'vwap':
        close = pd.read_csv(param['dailyinfo_close'], index_col=0)
        vwap = pd.read_csv(param['dailyinfo_vwap'], index_col=0)

    delist_date = pd.read_csv(param['basicinfo_delist_date'], index_col=0)
    delist_date = [pd.to_datetime(x) if type(x) == str else np.nan for x in delist_date.iloc[:, 0]]
    delist_date = pd.Series(delist_date, index=stock_code)

    # 月末权重转换到下月月初调仓
    raw_weight = pd.read_csv(param['optimal_weight'], index_col=0).loc[:, :param['bkt_end']]
    if list(daily_dates).index(pd.to_datetime(raw_weight.columns[-1])) + 1 == len(daily_dates):
        raw_weight = raw_weight.iloc[:, :-1]
    raw_weight.columns = [daily_dates[list(daily_dates).index(pd.to_datetime(x))+1].strftime('%Y-%m-%d') for x in raw_weight.columns]

    # 股票池补齐
    stock_weight = pd.DataFrame(0, index=stock_code, columns=raw_weight.columns)

    raw_weight = raw_weight.reindex(stock_code)
    stock_weight.loc[raw_weight.index,:] = raw_weight
    stock_weight = stock_weight.fillna(0)

    # 调仓日期
    stock_weight = stock_weight.loc[:, (np.array(stock_weight.columns >= param['bkt_start']) & np.array(stock_weight.columns <= param['bkt_end']))]
    dates_trade = pd.to_datetime(stock_weight.columns)

    # 回测初始化
    # 回测起止日期的序号
    id_date_bkt_start = np.where(daily_dates == pd.to_datetime(param['bkt_start']))[0][0]
    id_date_bkt_end = np.where(daily_dates == pd.to_datetime(param['bkt_end']))[0][0]

    # 换仓日的序号
    id_dates_trade = [np.where(daily_dates == i)[0][0] for i in dates_trade]

    # 储存净值
    value_daily_raw = pd.Series(np.nan, index=daily_dates)  # 组合净值
    value_daily_long = pd.Series(np.nan, index=daily_dates)  # 多头净值
    value_daily_short = pd.Series(np.nan, index=daily_dates)  # 空头净值

    # 第一个换仓日及上个交易日的净值置为1
    value_daily_raw.iloc[id_date_bkt_start-1:id_date_bkt_start+1] = 1
    value_daily_long.iloc[id_date_bkt_start-1:id_date_bkt_start+1] = 1
    value_daily_short.iloc[id_date_bkt_start-1:id_date_bkt_start+1] = 1

    # 储存上期权重，用于计算换手率；初始值置为0
    weight_last = pd.Series(0, index=stock_code)

    # 储存换手率
    turnover = pd.Series(0, index=dates_trade)

    # 停牌修正得到实际权重
    real_weight = pd.DataFrame(0, index=stock_weight.index, columns=pd.to_datetime(dates_trade).strftime('%Y-%m-%d'))

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
        if param['suspend_correct'] == 0:
            # 不进行修正
            real_weight.iloc[:, i_date_trade] = stock_weight.iloc[:, i_date_trade].values
        else:
            # 进行修正
            # 是否不可交易，1为不可交易
            flag_non_trade = ~((turn.iloc[:, id_date_trade] > 1e-4))
            # 是否退市，1为退市
            flag_delist = (delist_date <= daily_dates[id_date_trade])
            # 当期理论权重
            weight_now = stock_weight.iloc[:, i_date_trade]
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
                print('第%d期（%s）存在不可交易股票' % (i_date_trade + 1, dates_trade[i_date_trade]))
                # ------------------------------------------ #
                # Step1 上一期有持仓（多或者空），但是当期调仓日无法交易的，仓位保持不变
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
                        print('%s退市卖出' % (stock_code.iloc[i_stock]))

                # ------------------------------------------ #
                # Step2：上一期没有持仓，当期新增持仓但无法交易的，新增持仓置为0
                id_stock_non_trade2 = np.all([weight_now != 0, flag_non_trade == True], axis=0)
                weight_now.loc[id_stock_non_trade2] = 0
                # 输出新增持仓无法交易的股票
                if sum(id_stock_non_trade2) > 0:
                    id_stock_non_buy = np.where(id_stock_non_trade2 == True)[0]
                    for i_stock in id_stock_non_buy:
                        print('%s新增持仓无法交易' % (stock_code.iloc[i_stock]))

                # ------------------------------------------ #
                # Step3：实际当期真正权重 = 理论当期总权重 - 上一期无法交易的权重
                weight_now_real = weight_now_target - sum(real_weight.iloc[:, i_date_trade])
                # ------------------------------------------ #
                # Step4：修正当期权重，使得总仓位与目标仓位一致
                # if sum(weight_now) != 0:
                #     weight_now = weight_now * sum(weight_now) / weight_now_real  # todo:wrong?
                if sum(weight_now) != 0:
                    weight_now = weight_now * weight_now_real / sum(weight_now)

                # ------------------------------------------ #
                # Step5：买入
                id_stock_now = np.all([weight_now != 0, flag_non_trade == False], axis=0)
                id_stock_now = np.where(id_stock_now == True)[0]
                real_weight.iloc[id_stock_now, i_date_trade] = weight_now.iloc[id_stock_now]
                print('sum of weight target:{}'.format(weight_now_target))
                print('sum of weight now:{}'.format(real_weight.iloc[:, i_date_trade].sum()))

                """
                Step2~Step5从逻辑上来说是这样理解的：
                1）调仓日，上一期持仓里有的股票，把当天无法交易的股票单独摘出来，这部分股票所占的仓位（权重）不能动，剩余的原计划仓位（即weight_buy_real）进行各股票权重调整;
                2）在1）之后，上一期持仓里没有的股票，但新持仓要买入且当天无法交易的股票，新持仓置为0；
                3）在1）之后，上一期持仓里没有的股票，但新持仓要买入且当天可以交易，则对原计划仓位进行修正，即这部分股票占比要以除1）以外的原计划仓位为分母进行等比例修正，
                """

        # ------------------------------------------#
        # 3. 记录权重、买入价
        weight_now = real_weight.iloc[:, i_date_trade]
        if param['type_price'] == 'close':
            price_buy = close_adj.iloc[:, id_date_trade]
        elif param['type_price'] == 'vwap':
            # 以均价买入
            price_buy = vwap.iloc[:, id_date_trade] * close_adj.iloc[:, id_date_trade] / close.iloc[:, id_date_trade]
            # 若vwap为nan或0，则以前一日复权收盘价代替
            id_price_buy_nan = np.where(np.isnan(price_buy) == True)[0]
            id_price_buy_zero = np.where(price_buy == 0)[0]
            price_buy.iloc[id_price_buy_nan] = close_adj.iloc[id_price_buy_nan, id_date_trade - 1]
            price_buy.iloc[id_price_buy_zero] = close_adj.iloc[id_price_buy_zero, id_date_trade - 1]

        # 用于计算净值的基准value = 当期换仓日value
        # 当期换仓日value由上期换仓日的最后一个净值日根据卖出价计算得到
        # 当期换仓日value在随后计算时会根据收盘价更新
        value_port = value_daily_raw.iloc[id_date_trade]

        turnover.iloc[i_date_trade] = sum(abs(weight_now.values - weight_last.values))

        # 根据换手率扣除交易费用
        value_port = value_port * (1 - turnover.iloc[i_date_trade] * param['fee'])
        # ------------------------------------------#
        # 4. 计算每日净值
        for id_date_value in id_dates_value:
            if id_date_value < id_dates_value[-1]:
                # 非当期最后一个净值日
                price_value = close_adj.iloc[:, id_date_value]
            else:
                # 当期最后一个净值日，卖出
                if param['type_price'] == 'close':
                    price_value = close_adj.iloc[:, id_date_value]
                elif param['type_price'] == 'vwap':
                    # 以均价卖出
                    price_value = vwap.iloc[:, id_date_value] * close_adj.iloc[:, id_date_value] / close.iloc[:,id_date_value]
                    # 若vwap为nan或0，则以前一日复权收盘价代替
                    id_price_value_nan = np.where(np.isnan(price_value) == True)[0]
                    id_price_value_zero = np.where(price_value == 0)[0]
                    price_value.iloc[id_price_value_nan] = close_adj.iloc[id_price_value_nan, id_date_value - 1]
                    price_value.iloc[id_price_value_zero] = close_adj.iloc[id_price_value_zero, id_date_value - 1]

                # 计算自然增长的权重，用于计算换手率和交易费用
                weight_last = weight_now * price_value / price_buy
                weight_last.loc[np.isnan(weight_last)] = 0
                # 权重归一化，考虑现金仓位
                weight_last = weight_last / (sum(weight_last) + 1 - sum(weight_now))  # 这样归一化可以把杠杆也统一容纳进来
                print('sum of weight last:{}'.format(weight_last.sum()))

            # 计算收益
            returns_port = np.nansum(weight_now * (price_value / price_buy - 1))

            # 计算净值；若为当期换仓日的最后一个净值日，则此净值为下期换仓日计算净值的基准
            value_daily_raw.iloc[id_date_value] = value_port * (1 + returns_port)

    # 输出持仓
    real_weight = real_weight.loc[raw_weight.index,:]

    # 保存净值
    writer = pd.ExcelWriter(param['opt_bkt'])
    value_daily_raw = value_daily_raw.dropna()

    value_daily_raw.to_excel(writer, sheet_name='value_daily_port')
    real_weight.to_excel(writer, sheet_name='real_weight')
    turnover.to_excel(writer, sheet_name='turnover')
    writer.save()
    writer.close()

    return value_daily_raw


if __name__ == '__main__':
    param = init_param()
    param['optimal_weight'] = 'weight_sue_txt_layer1_equal.csv'
    value_daily_raw = BackTest(param)
    value_daily_raw.plot()

