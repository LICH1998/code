# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:29:04 2021

@author: chenwei
"""

import pandas as pd
import numpy as np


class Metrics:

    @staticmethod
    def annual_return(nav, annual=True):
        nav = nav.reset_index(drop=True)
        if annual:
            ret = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
        else:
            ret = (nav.iloc[-1] / nav.iloc[0]) - 1
        return ret

    @staticmethod
    def annual_vol(nav):
        nav = nav.reset_index(drop=True)
        vol = np.std(nav / nav.shift(1) - 1) * np.sqrt(252)
        return vol

    @staticmethod
    def sharpe_ratio(nav):
        sr = Metrics.annual_return(nav) / Metrics.annual_vol(nav)
        return sr

    @staticmethod
    def drawdown(nav):
        drawdown = []
        for i, value in enumerate(nav):
            drawdown.append(1 - (value / max(nav[:(i+1)])))

        if isinstance(nav, pd.Series):
            drawdown = pd.Series(drawdown, index=nav.index)

        return drawdown

    @staticmethod
    def max_drawdown(nav):
        drawdown = Metrics.drawdown(nav)
        return max(drawdown)

    @staticmethod
    def calmar_ratio(nav):
        calmar_ratio = Metrics.annual_return(nav) / Metrics.max_drawdown(nav)
        return calmar_ratio

    @staticmethod
    def excess_return(nav, nav_bench, annual=True):
        ra = nav / nav.shift(1) - 1
        rb = nav_bench / nav_bench.shift(1) - 1
        excess_nav = (1 + ra - rb).cumprod()
        excess_value = Metrics.annual_return(excess_nav.dropna(), annual=annual)

        return excess_value

    @staticmethod
    def excess_vol(nav, nav_bench):
        ra = nav / nav.shift(1) - 1
        rb = nav_bench / nav_bench.shift(1) - 1
        excess_nav = (1 + ra - rb).cumprod()
        excess_value = Metrics.annual_vol(excess_nav.dropna())

        return excess_value

    @staticmethod
    def excess_max_drawdown(nav, nav_bench):
        ra = nav / nav.shift(1) - 1
        rb = nav_bench / nav_bench.shift(1) - 1
        excess_nav = (1 + ra - rb).cumprod()
        excess_max_drawdown = Metrics.max_drawdown(excess_nav.dropna())

        return excess_max_drawdown

    @staticmethod
    def single_performance(nav, nav_bench, annual=True):
        if nav.index[0].year == nav.index[-1].year:
            year = nav.index[0].year
        else:
            year = '成立以来'

        performance = {
            '时间': year,
            '年化收益率': Metrics.annual_return(nav, annual=annual),
            '年化波动率': Metrics.annual_vol(nav),
            '最大回撤': Metrics.max_drawdown(nav),
            '夏普比率': Metrics.sharpe_ratio(nav),
            '卡玛比率': Metrics.calmar_ratio(nav),
            '年化超额收益': Metrics.excess_return(nav, nav_bench, annual=annual),
            '超额收益最大回撤': Metrics.excess_max_drawdown(nav, nav_bench)
        }
        return performance
