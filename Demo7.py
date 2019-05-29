# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
# 集成分类模型  使用不同的决策树进行模型训练和预测分析
# 1、数据读取
# 导入pandas用于数据分析。
import pandas as pd
# 利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据。
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

## 2、数据处理
# 机器学习有一个不太被初学者重视，并且耗时，但是十分重要的一环，特征的选择！这个需要基于一些背景知识。
# 根据我们对这场事故的了解，sex, age, pclass这些都很有可能是决定幸免与否的关键因素。
X=titanic[['pclass','age','sex']]
y=titanic['survived']

# 借由上面的输出，我们设计如下几个数据处理的任务：
# 1) age这个数据列，只有633个，需要补完。
# 2) sex 与 pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替。

# 首先我们补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略。
#inplace=True：不创建新的对象，直接对原始对象进行修改；
# inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
X['age'].fillna(X['age'].mean(), inplace=True)

#对原始数据进行分割，25%的乘客数据用于测试
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#对类别型特征进行转化，成为特征向量
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)  #sparse=False意思是不产生稀疏矩阵
#转换特征后，凡是类别型的特征都单独剥离出来，独成一列特征，数值型保持不变
X_train=vec.fit_transform(X_train.to_dict(orient='record'))

# 对测试数据的特征进行转换
X_test=vec.transform(X_test.to_dict(orient='record'))

# 一、使用单一决策树进行模型训练和预测分析
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()    #使用默认配置初始化决策树分类器
dtc.fit(X_train,y_train)    #使用分割得到的训练数据进行模型学习
dtc_y_predict=dtc.predict(X_test)    #使用训练好的决策树模型对测试特征数据进行预测

# 二、使用随机森林分类器进行集成模型的训练以及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)

# 三、使用梯度提升决策树进行集成模型的训练以及预测分析
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred=gbc.predict(X_test)

#集成模型对泰坦尼克号乘客是否生还的预测性能
from sklearn.metrics import classification_report

#输出单一决策树在测试集上的分类准确性，以及精确率、召回率、F1指标
print ('单一决策树的准确率是：',dtc.score(X_test,y_test))
print (classification_report(dtc_y_predict,y_test))

#输出随机森林分类器在测试集上的分类准确性，以及精确率、召回率、F1指标
print("随机分类器的准确率是:",rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test))

#输出梯度提升决策树在测试集上的分类准确性，以及精确率、召回率、F1指标
print('梯度提升决策树的准确性是：',gbc.score(X_test,y_test))
print(classification_report(gbc_y_pred,y_test))