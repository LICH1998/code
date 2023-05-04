#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-03-21 11:40
# @Author : lichangheng

import numpy as np
import pandas as pd

"""
data :列表
per:分位数，例如10分位数，4分位数
"""


def getPercentile(datas, per):
    final_array = pd.Series([per for i in range(len(datas))])
    per_num = 100 / per
    for i in range(1, per):
        percent = np.percentile(datas, (per - i) * per_num)
        final_array = np.where(datas < percent, (per - i), final_array)
    return final_array


array = [["001", "009", "006", "005", "007", "002", "004", "008", "003", "010"], [1, 4, 2, 3, 9, 8, 6, 5, 10, 7]]
data = pd.DataFrame(array, index=["code", "value"])
data = data.T

data["percent"] = getPercentile(data["value"], 4)
print(data)
