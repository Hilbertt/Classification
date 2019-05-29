# -*- coding:utf-8 -*-
# 决策树模型  需要参数训练，适用于非线性模型
# 1、数据读取
# 导入pandas用于数据分析。
import pandas as pd
# 利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据。
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 观察一下前几行数据，可以发现，数据种类各异，数值型、类别型，甚至还有缺失数据。
titanic.head()
# 使用pandas，数据都转入pandas独有的dataframe格式（二维数据表格），直接使用info()，查看数据的统计特性。
titanic.info()

## 2、数据处理
# 机器学习有一个不太被初学者重视，并且耗时，但是十分重要的一环，特征的选择！这个需要基于一些背景知识。
# 根据我们对这场事故的了解，sex, age, pclass这些都很有可能是决定幸免与否的关键因素。
X=titanic[['pclass','age','sex']]
y=titanic['survived']

#对当前的特征进行探查
X.info()

# 借由上面的输出，我们设计如下几个数据处理的任务：
# 1) age这个数据列，只有633个，需要补完。
# 2) sex 与 pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替。

# 首先我们补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略。
#inplace=True：不创建新的对象，直接对原始对象进行修改；
# inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
X['age'].fillna(X['age'].mean(), inplace=True)

#数据分割
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

### 3、分类预测

#使用sklearn.feature_extraction 中的特征转换器 DictVectorizer，详见3.1.1.1特征抽取
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)  #sparse=False意思是不产生稀疏矩阵

#转换特征后，凡是类别型的特征都单独剥离出来，独成一列特征，数值型保持不变
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
print (vec.feature_names_)  #查看转换后的列名

# 对测试数据的特征进行转换
X_test=vec.transform(X_test.to_dict(orient='record'))
#从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc=DecisionTreeClassifier()
#使用分割得到的训练数据进行模型学习
dtc.fit(X_train,y_train)
#使用训练好的决策树模型对测试特征数据进行预测
y_predict=dtc.predict(X_test)

#### 4、性能评估
from sklearn.metrics import classification_report
#输出预测准确性
print(dtc.score(X_test,y_test))
#输出更加详细的性能分析
print(classification_report(y_predict,y_test,target_names=['died','survived']))
