# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
#
# # 第一段
# # 定义要拟合的函数形式，这里使用对数函数 y = a * ln(x) + b + c*x
# def func1(x, a, b, c):
#     return a * np.log(x) + b + c*x
#
# # 定义输入数据
# x_data = np.array([0.2, 1, 6, 8, 10, 14, 16, 17.2])
# y_data = np.array([3, 7.26, 10.65, 11.33, 11.85, 12.41, 12.59, 12.67])
# # 定义输入数据
# # 使用 curve_fit 进行拟合
# popt, pcov = curve_fit(func1, x_data, y_data)
#
# # 输出拟合结果
# print("第一段拟合参数: a=%f, b=%f, c=%f" % (popt[0], popt[1], popt[2]))
#
# # 生成拟合曲线数据
# x_curve = np.linspace(0.2, 18, 200)
# y_curve = func1(x_curve, popt[0], popt[1], popt[2])
#
# # 绘制散点图和拟合曲线图
# plt.scatter(x_data, y_data)
# plt.plot(x_curve, y_curve, 'r')
# plt.title('the first curve')
# plt.show()
#
# # 第二段
# # 定义要拟合的函数形式，这里使用对数函数 y = a * ln(x) + b + c*x
# def func2(x, a, b, c):
#     return a * np.log(x) + b + c*x**2
#
# x_data = np.array([17.2,22,27.96,34.02,39.96])
# y_data = np.array([12.67,14.7,16.64,18.29,18.81])
# # 使用 curve_fit 进行拟合
# popt, pcov = curve_fit(func2, x_data, y_data)
#
# # 输出拟合结果
# print("第二段拟合参数: a=%f, b=%f, c=%f" % (popt[0], popt[1], popt[2]))
#
# # 生成拟合曲线数据
# x_curve = np.linspace(17, 40, 200)
# y_curve = func2(x_curve, popt[0], popt[1], popt[2])
#
# # 绘制散点图和拟合曲线图
# plt.scatter(x_data, y_data)
# plt.plot(x_curve, y_curve, 'r')
# plt.title('the second curve')
# plt.show()
#
# # 第三段
# # 定义要拟合的函数形式，这里使用对数函数 y = a * ln(x) + b + c*x
# def func3(x, a, b, c):
#     return a * np.log(x) + b + c*x
#
# x_data = np.array([39.96,44,49.97,56,59.96,63,70.8,75.93])
# y_data = np.array([18.81,18.8,18.35,17.42,16.75,16,13.98,12.5])
#
# # 使用 curve_fit 进行拟合
# popt, pcov = curve_fit(func3, x_data, y_data)
#
# # 输出拟合结果
# print("第三段拟合参数: a=%f, b=%f, c=%f" % (popt[0], popt[1], popt[2]))
#
# # 生成拟合曲线数据
# x_curve = np.linspace(40, 80, 200)
# y_curve = func3(x_curve, popt[0], popt[1], popt[2])
#
# # 绘制散点图和拟合曲线图
# plt.scatter(x_data, y_data)
# plt.plot(x_curve, y_curve, 'r')
# plt.title('the third curve')
# plt.show()
#
#
#
#
# import pandas as pd
# import numpy as np
#
# company = ['A', 'B', 'C']
# data = pd.DataFrame({
#     "company": [company[x] for x in np.random.randint(0, len(company), 10)],
#     "salary": np.random.randint(5, 50, 10),
#     "age": np.random.randint(15, 50, 10)
# }
# )
#
# group = data.groupby("company")
#
# def l(data):
#     return data['salary'] + data['age']
#
# group['test'] = group['age'].apply(l, axis=1)
# print(a)
import numpy as np
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# estimator = PCA(n_components=20)
# X_train = np.random.randn(10000).reshape(1000,10)
# # pca_X_train = estimator.fit_transform(X_train)
# # print(X_train.shape)
# data = pd.DataFrame(X_train)
# # print(data.head())
# # print(data.corr())
# # 画热力图
# import seaborn as sns
# ax = sns.heatmap(data.corr(), annot=True, cmap='RdBu_r', xticklabels=1, yticklabels=1)
# ax.plot()
# plt.show()


# import torch
# import random
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
# model = model.eval()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)


# 导入必要的库
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
#
# # 生成数据
# X, y = make_blobs(n_samples=1000, centers=4, random_state=42)
# print(X.shape)
#
# # 实例化KMeans模型
# kmeans = KMeans(n_clusters=4)
#
# # 拟合模型
# kmeans.fit(X)
#
# # 预测聚类标签
# labels = kmeans.predict(X)
#
# # 可视化聚类结果
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
# plt.show()

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate some random data for clustering
# data = np.random.randn(1000, 10)
#
# # Split the data into training and testing sets
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#
# # Create a KMeans object and fit the training data
# kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
# kmeans.fit(train_data)
#
# # Get the labels for each data point in the test set
# test_labels = kmeans.predict(test_data)
#
# # Perform PCA to reduce the data to 2 dimensions
# pca = PCA(n_components=2)
# pca.fit(data)
# test_data_pca = pca.transform(test_data)
#
# # Plot the test data with different colors for each predicted cluster
# plt.figure(figsize=(10, 8))
# plt.scatter(test_data_pca[:, 0], test_data_pca[:, 1], c=test_labels, cmap='rainbow')
# plt.title('KMeans Clustering Results')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()


# # 招商银行的文件
# import pandas as pd
#
# # 读取txt文件
# with open('./招商data/data_ts.txt', 'r') as f:
#     lines = f.readlines()
#
# # 对每一行进行处理
# results = []
# column = lines[0].strip().split('\t')
# print(column)
#
# for line in lines[1:]:
#     # 这里可以添加对每行的处理逻辑
#     processed_line = line.strip().split('\t')
#     result = [x for x in processed_line][1:]  # 举例：将每行以逗号分隔的字符串转化为整数列表
#     results.append(result)
#
# data = pd.DataFrame(results,columns=column).sort_values(by='cust_wid',axis=0).reset_index(drop=True)
# print(data.head())
# data.to_csv('./招商data/data_ts.csv',index=False)

# import numpy as np
# a = np.array([2,4,6,8,10])
# b = np.array([2,4,6,8,10])
# #只有一个参数表示条件的时候
# s = np.where((a<5)&(a>1))
# s = (a<5)
# print(s)

# import numpy as np
# a = [True,False]
# b = [False, False]
# c = np.array(a)&np.array(b)
# print(c)

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# dfs = pd.read_excel('./行业轮动/industry.xlsx', sheet_name=['week', 'month'])
# df1 = dfs.get('week').set_index('行业名称').drop(['指数代码'], axis=1).T
# print(df1.iloc[:5])
# a = df1.corr()
# print(a)
# res1 = []
# res2 = []
# for i in a.index:
#     l1 = a[i].nlargest(2).min()
#     l2 = a[i].nlargest(2).idxmin()
#     l3 = a[i].nlargest(3).min()
#     l4 = a[i].nlargest(3).idxmin()
#     res1.append((i,l1,l2))
#     res2.append((i,l3,l4))
# print(res1)
# print(res2)
#
# # 画热力图
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# plt.rcParams['axes.unicode_minus'] = False    # 解决无法显示符号的问题
# sns.set(font='SimHei', font_scale=0.8)        # 解决Seaborn中文显示问题
# ax = sns.heatmap(a, annot=False,linewidths=0.1,linecolor="grey", cmap='RdBu_r')
# ax.plot()
# plt.show()

# import numpy as np
# a = np.arange(0, 1.0, 0.1)
# print(a)
#
# import pandas as pd
# df = pd.DataFrame({'a': [3.0, 2.0, 4.0, 1.0],'b': [1.0, 4.0 , 2.0, 3.0]})
# print(df)
# print(df.reindex([1,2]))

# import pandas as pd
#
# def normalize_columns(df):
#     normalized_df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#     return normalized_df
#
# # 示例用法
# data = {'col1': [2, 5, 10, 8, 3], 'col2': [1, 6, 9, 4, 7]}
# df = pd.DataFrame(data)
# normalized_df = normalize_columns(df)
# print(normalized_df)

# import pandas as pd
#
# data = {'col1': ['apple', 'orange', 'banana', 'apple', 'grape']}
# df = pd.DataFrame(data)
#
# # 替换特定字符串（例如，'apple'）为空值
# df = df.replace('apple', np.NaN).fillna('100')
#
# print(df)

# import pandas as pd
# import numpy as np
#
# a = [1,2,3]
# b = [1,2,3]
# c = []
# c.append(a)
# c.append(b)
# c = np.array(c).T
# d = pd.DataFrame(c,columns=['1','2'])
# print(d)

a = ['1','2','2']
a = a + ['3']
print(a)
