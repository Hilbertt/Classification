# -*- coding:utf-8 -*-
# K近邻分类 属于无参数模型，不通过学习算法分析训练数据，根据训练数据的分布直接作出分类决策，算法复杂度高
# 1、数据读取
#从sklearn.datasets导入iris数据加载器
from sklearn.datasets import load_iris
#使用加载器读取数据并存入变量iris
iris=load_iris()
#检查数据规模和数据说明
print(iris.data.shape)
print(iris.DESCR)

## 2、数据处理
#对iris数据集进行分割
#从from sklearn.model_selection 里导入train_test_split用于数据分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

### 3、分类预测
#从sklearn.preprocessing中导入数据标准化模块StandardScaler
from sklearn.preprocessing import StandardScaler
#从sklearn.neighbors 里选择导入K近邻分类器 KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#对训练和测试的特征数据进行标准化
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

#使用K近邻分类器对测试数据进行类别预测，预测结果存储在变量y_predict中
knc=KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict=knc.predict(X_test)

#### 4、性能评估
#使用模型自带的评估函数进行准确性测评
print("K近邻分类器的准确率是：",knc.score(X_test,y_test))

#依然使用sklearn.metrics里面的classification_report模块对预测结果做详细的分析
#sklearn中的classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names)) 
