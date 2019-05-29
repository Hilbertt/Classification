# -*- coding:utf-8 -*-
# Filename : 良性、恶性乳腺癌肿瘤预测 (线性分类器)
#导入pandas工具包并更名为pd
import pandas as pd

#调用pandas工具包的 read_csv函数,导入训练文件的地址参数,获得返回的数据并保存在 df_train
df_train = pd.read_csv(r'C:\Users\jianliu\Desktop\Code\Data\breast-cancer-train.csv')
df_test = pd.read_csv(r'C:\Users\jianliu\Desktop\Code\Data\breast-cancer-test.csv')
#别的写法:df_train = pd.read_csv('C:\\Users\\jianliu\\Desktop\\python\\Code\\breast-cancer-train.csv')
#或者(r'C:\Users\jianliu\Desktop\python\Code\breast-cancer-train.csv')
#抑或(r'C:/Users/jianliu/Desktop/python/Code/breast-cancer-test.csv')

# 选取'Clump Tickness'和 'Cell Size'作为特征,构建测试集的正负分类样本,loc——通过行标签索引行数据
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

#导入画图工具并更名为 plt
import matplotlib.pyplot as plt

#绘制良性肿瘤样本点,标记为红;绘制恶性肿瘤样本点,标记为黑
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')

#对XY轴进行说明
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

plt.show()

import numpy as np
#利用numpy中的random函数随机采样直线的截距和系数
intercept = np.random.random([1])    #随机生成[0,1)之间的一维随机数
coef = np.random.random([2])    #随机生成[0,1)之间的二维随机数

lx = np.arange(0, 12)    #生成0到11的数列，步长默认为1
ly = (-intercept - lx * coef[0]) / coef[1]
#绘制一条随机直线
plt.plot(lx, ly, c='yellow')

plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()    #显示图像

#导入sklearn中的逻辑斯谛回归分类器
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()    #调用逻辑斯谛函数，lr是一个实例对象

#使用前十条训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print ('Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

intercept = lr.intercept_    #intercept_和coef_都是逻辑斯谛模型的参数，截距intercept是一维的，系数coef有多个。
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

#使用所有的训练样本学习直线的系数和截距
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print ('Testing accuracy (all training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()