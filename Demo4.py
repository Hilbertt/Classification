# -*- coding：utf-8 -*-
#朴素贝叶斯分类模型(基于贝叶斯理论) 假设特征条件独立，节约内存和时间，对于关联较强的分类任务，性能表现不佳
# 适用于文本分类任务，如互联网新闻的分类，垃圾邮件的筛选
# 1、读取数据
#从sklearn.datasets中导入新闻数据抓取器 fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
#fetch_20newsgroups需要从网上下载数据，并存储为news
news=fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])

## 2、数据处理
#从sklearn.model_selection导入train_test_split
from sklearn.model_selection import train_test_split
#随机采样25%的数据样本作为测试集，随机数种子为33
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)

### 3、分类预测
#下面的工作，首先将文本转化为特征向量，然后估计参数，然后对同样转化为特征向量的样本进行类别预测。
# 从sklearn.feature_extraction.txt 里导入文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()    #实例化对象
X_train=vec.fit_transform(X_train)    #得到训练集的特征向量，fit模型参数，再做归一化处理
X_test=vec.transform(X_test)

#从sklearn.naive_bayes 里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
#使用默认配置初始化贝叶斯朴素模型
mnb=MultinomialNB()
mnb.fit(X_train,y_train)    #利用训练数据对模型参数进行估计
y_predict=mnb.predict(X_test)    #对测试样本进行类别预测和存储

#### 4、性能评估
# 从sklearn.metrics 中导入classification_report用于分类性能报告
from sklearn.metrics import classification_report
print('朴素贝叶斯的准确率是：',mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))



