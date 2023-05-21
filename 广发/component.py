#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-05-19 0:43
# @Author : lichangheng

from typing import List

import pandas as pd
import requests
from tqdm import tqdm


def get_public_dates(fund_code: str) -> List[str]:
    """
    获取历史上更新持仓情况的日期列表

    Parameters
    ----------
    fund_code : str
        6 位基金代码

    Returns
    -------
    List[str]
        指定基金公开持仓的日期列表

    """

    params = (
        ('FCODE', fund_code),
        ('OSVersion', '14.3'),
        ('appVersion', '6.3.8'),
        ('deviceid', '3EA024C2-7F22-408B-95E4-383D38160FB3'),
        ('plat', 'Iphone'),
        ('product', 'EFund'),
        ('serverVersion', '6.3.6'),
        ('version', '6.3.8'),
    )
    url = 'https://fundmobapi.eastmoney.com/FundMNewApi/FundMNIVInfoMultiple'
    headers = {
        'User-Agent': 'EMProjJijin/6.2.8 (iPhone; iOS 13.6; Scale/2.00)',
        'GTOKEN': '98B423068C1F4DEF9842F82ADF08C5db',
        'clientInfo': 'ttjj-iPhone10,1-iOS-iOS13.6',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Host': 'fundmobapi.eastmoney.com',
        'Referer': 'https://mpservice.com/516939c37bdb4ba2b1138c50cf69a2e1/release/pages/FundHistoryNetWorth',
    }
    json_response = requests.get(
        url,
        headers=headers,
        params=params).json()
    if json_response['Datas'] is None:
        return []
    return json_response['Datas']


def get_inverst_postion(code: str, date=None) -> pd.DataFrame:
    '''
    根据基金代码跟日期获取基金持仓信息
    -
    参数

        code 基金代码
        date 公布日期 形如 '2020-09-31' 默认为 None，得到最新公布的数据
    返回

        持仓信息表格

    '''
    EastmoneyFundHeaders = {
        'User-Agent': 'EMProjJijin/6.2.8 (iPhone; iOS 13.6; Scale/2.00)',
        'GTOKEN': '98B423068C1F4DEF9842F82ADF08C5db',
        'clientInfo': 'ttjj-iPhone10,1-iOS-iOS13.6',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Host': 'fundmobapi.eastmoney.com',
        'Referer': 'https://mpservice.com/516939c37bdb4ba2b1138c50cf69a2e1/release/pages/FundHistoryNetWorth',
    }
    params = [
        ('FCODE', code),
        ('MobileKey', '3EA024C2-7F22-408B-95E4-383D38160FB3'),
        ('OSVersion', '14.3'),
        ('appType', 'ttjj'),
        ('appVersion', '6.2.8'),
        ('deviceid', '3EA024C2-7F22-408B-95E4-383D38160FB3'),
        ('plat', 'Iphone'),
        ('product', 'EFund'),
        ('serverVersion', '6.2.8'),
        ('version', '6.2.8'),
    ]
    if date is not None:
        params.append(('DATE', date))
    params = tuple(params)

    response = requests.get('https://fundmobapi.eastmoney.com/FundMNewApi/FundMNInverstPosition',
                            headers=EastmoneyFundHeaders, params=params)
    rows = []
    stocks = response.json()['Datas']['fundStocks']

    columns = {
        'GPDM': '股票代码',
        'GPJC': '股票简称',
        'JZBL': '持仓占比(%)',
        'PCTNVCHG': '较上期变化(%)',
    }
    if stocks is None:
        return pd.DataFrame(rows, columns=columns.values())

    df = pd.DataFrame(stocks)
    df = df[list(columns.keys())].rename(columns=columns)
    return df


if __name__ == "__main__":
    data = pd.read_excel('./fund_code_list.xlsx', dtype=str)
    code_list = data['基金代码'].tolist()
    # 存储重仓股代码
    stock_code_list = []
    stock_name_list = []
    for i in tqdm(range(len(code_list))):
        code = code_list[i]
        # 6 位基金代码
        # 创建 excel 文件
        writer = pd.ExcelWriter(f'./component/{code}.xlsx')
        try:
            # 获取基金公开持仓日期
            public_dates = get_public_dates(code)
            # 遍历全部公开日期，获取该日期公开的持仓信息
            date = public_dates[-1]
            print(f'正在获取 {date} 的持仓信息......')
            df = get_inverst_postion(code, date=date)
            stock_code_list.extend(df['股票代码'].tolist())
            stock_name_list.extend(df['股票简称'].tolist())
            # 添加到 excel 表格中
            df.to_excel(writer, index=None, sheet_name=date)
            print(f'{date} 的持仓信息获取成功')
            writer.save()
            print(f'{code} 的历史持仓信息已保存到文件 {code}.xlsx 中')
        except:
            print(f'{date} 的持仓信息获取失败')
            continue
    res = pd.DataFrame(list(zip(stock_code_list, stock_name_list)), columns=['code', 'name'])
    res.drop_duplicates(subset=['code', 'name'], keep='first', inplace=True)
    def func(x):
        if len(x) == 6 and x.isdigit():
            return True
        else:
            return False

    res['mark'] = res['code'].apply(func)
    res = res[res['mark']==True].reset_index(drop=True).drop(columns=['mark'])
    res.to_csv('./stock_code.csv',index=False,encoding='utf-8')
