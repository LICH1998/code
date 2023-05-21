### --py文件:

component.py  根据基金代码获取基金的重仓股并汇总

data_process.py 提取重仓股的特征并汇总

cluster.py 根据重仓股的特征进行聚类

### --csv文件:

fund_code_list.xlsx 广发旗下基金代码

stock_code.csv 重仓股的代码

data_total.csv 重仓股对应的特征

turn.csv 重仓股对应的换手率范围

cluster_res.csv 重仓股分类对应的类别

### --dir:

component 基金以及对应的重仓股

stock_component 重仓股对应的指标


### 运行：
python component.py

python data_process.py

python cluster.py
