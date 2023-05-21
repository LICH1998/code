# -*- coding: utf-8 -*-
"""
@author: HTSC
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import joblib
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

def init_param():
    """
    初始化参数
    >>> param = init_param()
    """
    # 策略参数 ----------
    param = dict()
    param['train_len'] = 6  # 以过去6个月作为样本内
    # param['num_titlekeywords'] = 200  # 标题关键词筛选个数
    # param['num_contentkeywords'] = 1000  # 摘要关键词筛选个数
    param['ls_month_test'] = [[129, 140], [141, 152], [153, 164], [165, 176], [177, 188],
                              [189, 200], [201, 212], [213, 224], [225, 236], [237, 248],
                              [249, 260], [261, 272], [273, 284], [285, 291]]

    # 逻辑回归参数
    param['ls_logi_c'] = [1e-5, 3e-5, 6e-5, 8e-5,
                          0.0001, 0.0003, 0.0006, 0.0008,
                          0.001, 0.003, 0.006, 0.008, 0.01]

    # XGBoost参数
    param['ls_xgbc_learning_rate'] = [0.025, 0.05, 0.075]
    param['ls_xgbc_max_depth'] = [3, 5]
    param['ls_xgbc_subsample'] = [0.8, 0.85, 0.9, 0.95]

    # LightGBM参数
    param['ls_lgbc_learning_rate'] = [0.025, 0.05, 0.075]
    param['ls_lgbc_max_depth'] = [3, 5, 7]
    param['ls_lgbc_feature_fraction'] = [0.8, 0.9, 1]

    # 随机森林参数
    param['ls_rdfc_n_estimators'] = [100, 200, 300]
    param['ls_rdfc_max_depth'] = [5, 7, 9]

    # GBDT参数
    param['ls_gbdt_learning_rate'] = [0.001, 0.01, 0.1]
    param['ls_gbdt_subsample'] = [0.8, 0.85, 0.9]
    param['ls_gbdt_max_depth'] = [3, 5]

    # SVC参数
    param['ls_svc_c'] = [0.5, 1, 2]

    # 其他参数
    param['seed'] = 42
    param['k_fold'] = 5
    param['model'] = 'XGBC'  # Union['LOGI', 'XGBC', 'LGBC', 'RDFC', 'GBDT', 'SVC']

    # 路径参数 ----------
    param['dailyinfo_dates'] = '../raw_data/general_data/day.csv'  # 日频日期序列
    param['monthlyinfo_dates'] = '../raw_data/general_data/month.csv'  # 月频日期序列

    # 财报和业绩预告和业绩快报取较早
    param['data_match_plus_AR_splitword'] = '../data/data_report_adjust_split_word_allinfo.csv'
    param['path_model'] = f"../model/XGBC/{param['model']}/"
    param['suffix'] = f"_roll_{param['model']}_kfold"
    param['path_factor'] = '../result/raw_factor.csv'

    return param


class TrainTest:
    def __init__(self):
        # 通用变量
        self.para = init_param()
        # 读取上一步中计算好的结果
        self.raw_data = pd.read_csv(self.para['data_match_plus_AR_splitword'], index_col=0, dtype={'STOCK_CODE': 'str'})
        # 读取交易日期和月份
        self.monthly_dates = pd.read_csv(self.para['monthlyinfo_dates'], index_col=0)['0'].tolist()
        self.daily_dates = pd.read_csv(self.para['dailyinfo_dates'], index_col=0)['0'].tolist()

        # 局部变量：存储每次滚动训练时词名
        self.title_feature = None
        self.content_feature = None

        # 存储最终结果的因子
        self.factor = pd.DataFrame()

    def main_train(self):
        """主训练函数"""
        for i_year in range(len(self.para['ls_month_test'])):
            print("*" * 100)
            print("Training round: %d/%d" % (i_year + 1, len(self.para['ls_month_test'])))
            id_month_test = self.para['ls_month_test'][i_year]  # 测试集：未来一年
            id_month_train = [id_month_test[0] - self.para['train_len'], id_month_test[0] - 1]  # 训练集：过去两年

            # 提取样本内
            month_train = self.monthly_dates[id_month_train[0] - 1: id_month_train[1] + 1]  # 截取训练周期
            date_train_start = month_train[0]
            # 训练日期的结束是month_train的最后一个日期的前五个交易日（防止信息泄露）
            date_train_end = self.daily_dates[self.daily_dates.index(month_train[-1]) - 5]

            # 截取训练集
            self.train_data = self.raw_data[
                (self.raw_data['REPORT_DATE'] > date_train_start) & (self.raw_data['REPORT_DATE'] <= date_train_end)]

            # 处理格式
            CONTENT_processed = self.train_data['CONTENT_processed'].tolist()
            CONTENT_processed = pd.Series(CONTENT_processed).fillna('').tolist()
            CONTENT_processed = [n.lstrip('[').rstrip(']').split(',') for n in CONTENT_processed]
            CONTENT_processed = [[float(m) for m in n] for n in CONTENT_processed]
            X_in_sample = np.matrix(CONTENT_processed)

            quantile_low = np.quantile(self.train_data['AR'], 0.3)
            quantile_high = np.quantile(self.train_data['AR'], 0.7)
            # 打收益标签（其中self.label是本对象功能函数）
            y_in_sample = [self.label(ar, quantile_low, quantile_high) for ar in self.train_data['AR']]

            # 交叉验证训练模型
            if self.para['model'] == 'LOGI':
                model = linear_model.LogisticRegression(random_state=self.para['seed'], penalty='elasticnet',
                                                        l1_ratio=0.5, solver='saga', max_iter=200)
                param_grid = [{'C': self.para['ls_logi_c']}]

            elif self.para['model'] == 'XGBC':
                model = xgb.XGBClassifier(random_state=self.para['seed'], n_jobs=-1, tree_method='gpu_hist', gpu_id=0,
                                          use_label_encoder=False, eval_metric=['mlogloss'])  # GPU
                # model = xgb.XGBClassifier(random_state=self.para['seed'], n_jobs=-1, tree_method='hist', use_label_encoder=False)  # CPU
                param_grid = [{'learning_rate': self.para['ls_xgbc_learning_rate'],
                               'max_depth': self.para['ls_xgbc_max_depth'],
                               'subsample': self.para['ls_xgbc_subsample']}]

            elif self.para['model'] == 'LGBC':
                model = lgb.LGBMClassifier(random_state=self.para['seed'], n_jobs=-1)
                param_grid = [{'learning_rate': self.para['ls_lgbc_learning_rate'],
                               'max_depth': self.para['ls_lgbc_max_depth'],
                               'feature_fraction': self.para['ls_lgbc_feature_fraction']}]

            elif self.para['model'] == 'RDFC':
                model = RandomForestClassifier(random_state=self.para['seed'], n_jobs=-1)
                param_grid = [{'n_estimators': self.para['ls_rdfc_n_estimators'],
                               'max_depth': self.para['ls_rdfc_max_depth']}]

            elif self.para['model'] == 'GBDT':
                model = GradientBoostingClassifier(random_state=self.para['seed'], n_estimators=10)
                param_grid = [{'learning_rate': self.para['ls_gbdt_learning_rate'],
                               'subsample': self.para['ls_gbdt_subsample'],
                               'max_depth': self.para['ls_gbdt_max_depth']}]

            # kfold交叉验证方法设定
            cv_method = KFold(n_splits=self.para['k_fold'], shuffle=True, random_state=self.para['seed'])

            # 模型训练
            model_cv = GridSearchCV(model, param_grid, cv=cv_method, verbose=10)
            model_cv.fit(X_in_sample, y_in_sample)

            # 最优模型保存
            model_cv.cv = None
            joblib.dump(model_cv, self.para['path_model'] + 'model_cv_' + str(i_year) + self.para['suffix'] + '.m')
            
            # 在有些Xgboost版本下无法直接保存XGBC的模型
            # if self.para['model'] == 'XGBC':
            #     model_best = model_cv.best_estimator_
            #     model_best.save_model(self.para['path_model'] + 'model_best_' + str(i_year) + self.para['suffix'] + '.bin')

    def main_test(self):
        """主预测函数"""
        for i_year in range(len(self.para['ls_month_test'])):
            print("*" * 100)
            print("Test round: %d/%d" % (i_year + 1, len(self.para['ls_month_test'])))
            id_month_test = self.para['ls_month_test'][i_year]
            month_test = self.monthly_dates[id_month_test[0] - 1: id_month_test[1] + 1]

            # 读取测试集
            date_test_start = month_test[0]
            date_test_end = month_test[-1]

            X_curr = self.raw_data[
                (self.raw_data['REPORT_DATE'] > date_test_start) & (self.raw_data['REPORT_DATE'] <= date_test_end)]
            print(X_curr)

            CONTENT_processed_curr = pd.Series(X_curr['CONTENT_processed']).fillna('').tolist()
            CONTENT_processed_curr = [n.lstrip('[').rstrip(']').split(',') for n in CONTENT_processed_curr]
            CONTENT_processed_curr = [[float(m) for m in n] for n in CONTENT_processed_curr]
            X_outof_sample = np.matrix(CONTENT_processed_curr)

            # 读取交叉验证最优模型 (在有些Xgboost版本下无法直接保存XGBC的模型)
            # if self.para['model'] in ['LOGI', 'LGBC', 'GBDT', 'RDFC', 'SVC']:
            #     model_cv = joblib.load(self.para['path_model'] + 'model_cv_' + str(i_year) + self.para['suffix'] + '.m')
            #     model_best = model_cv.best_estimator_
            # elif self.para['model'] == 'XGBC':
            #     model_best = xgb.XGBClassifier(eval_metric=['mlogloss'])
            #     model_best.load_model(self.para['path_model'] + 'model_best_' + str(i_year) + self.para['suffix'] + '.bin')

            model_cv= joblib.load(self.para['path_model'] + 'model_cv_' + str(i_year) + self.para['suffix'] + '.m')
            score_curr = model_cv.best_estimator_.predict_proba(X_outof_sample)

            # 计算因子
            factor = self.calc_factor(score_curr)
            self.factor = self.factor.append(pd.DataFrame({'STOCK_CODE': X_curr['STOCK_CODE'].values,
                                                           'REPORT_DATE': X_curr['REPORT_DATE'].values,
                                                           'CONTENT': X_curr['CONTENT'].values,
                                                           'factor': factor.reshape(-1)}))
            print(len(factor))

        # 保存结果
        self.factor = self.factor.reset_index(drop=True)
        self.factor.to_csv(self.para['path_factor'], encoding='utf_8_sig')

    # 对AR进行标签
    def label(self, x, quantile_low, quantile_high):

        if x > quantile_high:
            return 2
        if x < quantile_low:
            return 0
        else:
            return 1

    # 依据论文方法，计算SUE的值
    def calc_factor(self, data_proba):
        log_odds_low = np.log2(data_proba[:, 0] / (1 - data_proba[:, 0])).reshape(-1, 1)
        log_odds_high = np.log2(data_proba[:, 2] / (1 - data_proba[:, 2])).reshape(-1, 1)
        return log_odds_high - log_odds_low


if __name__ == '__main__':
    train_test = TrainTest()
    train_test.main_train()
    train_test.main_test()