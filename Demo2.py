# -*- coding:utf-8 -*-
# 线性分类模型(基于线性假设)  假设特征与分类结果存在线性关系的模型，通过累加各个维度的特征与各自权重的乘积进行类别决策
# 需要学习模型参数，模型包括LogisticRegression和SGDClassifier，前者耗时长性能高，后者耗时短性能低
# 1、良/恶性乳腺肿瘤数据预处理
# 导入pandas与numpy工具包
import pandas as pd
import numpy as np

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# 使用pandas.read_csv函数从互联网读取指定数据。
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names = column_names )

# 将?替换为标准缺失值表示。
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
data = data.dropna(how='any')

# 输出data的数据量和维度。
print(data.shape)

## 2、准备训练、测试数据

# 使用sklearn.cross_valiation里的train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split

# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
#其中test_size是样本占比，random_state是随机数种子，也就是说，从683行数据中随机抽取0.75作为训练集，0.25作为测试集
#且1到9列是x的元素，第10列是y的元素

#检查训练、测试样本的数量和类别分类
print(y_train.value_counts())
print(y_test.value_counts())

### 3、利用线性分类模型进行肿瘤预测任务
from sklearn.preprocessing import StandardScaler    #标准化数据
from sklearn.linear_model import LogisticRegression    #逻辑斯谛模型
from sklearn.linear_model import SGDClassifier    #随即梯度上升

#标准化数据，保证每个维度的特征数据的方差为1，均值为0
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

# 初始化LogisticRegression与SGDClassifier。
lr = LogisticRegression()
sgdc = SGDClassifier()

# 调用LogisticRegression中的fit函数/模块用来训练模型参数。
lr.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果储存在变量lr_y_predict中。
lr_y_predict = lr.predict(X_test)

# 调用SGDClassifier中的fit函数/模块用来训练模型参数。
sgdc.fit(X_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果储存在变量sgdc_y_predict中。
sgdc_y_predict = sgdc.predict(X_test)

#### 4、利用线性分类模型进行肿瘤预测任务的性能分析
# 从sklearn.metrics里导入classification_report模块。
from sklearn.metrics import classification_report

# 使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果。
print ('Accuracy of LR Classifier:', lr.score(X_test, y_test))
# 利用classification_report模块获得LogisticRegression其他三个指标的结果。
print (classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

# 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果。
print ('Accuarcy of SGD Classifier:', sgdc.score(X_test, y_test))
# 利用classification_report模块获得SGDClassifier其他三个指标的结果。
print (classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))


