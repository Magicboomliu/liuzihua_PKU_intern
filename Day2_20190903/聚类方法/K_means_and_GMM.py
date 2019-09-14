__author__ = "Luke Liu"
#encoding="utf-8"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

'''  构建K-Means模型  '''
iris = load_iris()
iris_data = iris['data'] # 提取数据集中的数据
iris_target = iris['target'] # 提取数据集中的标签
iris_names = iris['feature_names'] # 提取特征名
print(len(iris_names))
#直接数据归一化
scale = MinMaxScaler().fit(iris_data) # 训练规则
iris_dataScale = scale.transform(iris_data) # 应用规则

# 建立一个聚类为4的点
kmeans = KMeans(n_clusters=8,random_state=1234).fit(iris_dataScale) # 构建并训练模型
# print('构建的K-Means模型为：\n',kmeans)
# result = kmeans.predict([[1.5,1.5,1.5,1.5]])
# print('花瓣花萼长度宽度全为1.5的鸢尾花预测类别为：',result[0])
# print(kmeans.score(iris_data,iris_target))


'''  聚类结果可视化  '''
tsne = TSNE(n_components=2,init='random',random_state=177).fit(iris_data)    # 使用TSNE进行数据降维，降成两维
df = pd.DataFrame(tsne.embedding_)                    # 将原始数据转换为DataFrame
df['labels'] = kmeans.labels_     # 将聚类结果存储进df数据表中
df1 = df[df['labels']==0]
print(df1)
df2 = df[df['labels']==1]
df3 = df[df['labels']==2]
df4 =df[df['labels']==3]
df5 = df[df['labels']==4]
df6 = df[df['labels']==5]
df7 = df[df['labels']==6]
df8 =df[df['labels']==7]
# fig = plt.figure(figsize=(9,6))    # 绘制图形  设定空白画布，并制定大小
plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD',df4[0],df4[1],'yo',df5[0],df5[1],'ro',df6[0],df6[1],'b*',df7[0],df7[1],'gD',df8[0],df8[1],'go')
plt.show()                          # 显示图片