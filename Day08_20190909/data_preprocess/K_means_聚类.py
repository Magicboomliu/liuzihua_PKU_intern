__author__ = "Luke Liu"
#encoding="utf-8"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import  os
import numpy as np
from PIL import  Image
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
paths_labels=unpickle("AM_CGAN")
images_path = paths_labels['images_path']
redefined_labels=paths_labels['scores_path']

index=[0,1,2,3,4,5,6,7]
score_rank=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5]
index_scores_dict=dict(list(zip(score_rank,index)))
redefined_labels_index=[]
for i in redefined_labels:
    redefined_labels_index.append(index_scores_dict[i])
#处理image
images_array=np.zeros((2000,64,64,3),dtype=np.float32)
dataset_path='D:\BaiduYunDownload\python_exe\dataset\scut_faces\Images'
images_paths=[os.path.join(dataset_path,i) for i in images_path]
for i in range(2000):
    img=Image.open(images_paths[i])
    img=img.resize((64,64))
    img=np.asarray(img)
    im=img/255.
    images_array[0,:,:,:]+=im
    print("finis {}/2000".format(i+1))
print(len(list(set(redefined_labels_index))))
images_array=np.reshape(images_array,(2000,64*64*3))
#直接数据归一化
scale = MinMaxScaler().fit(images_array) # 训练规则
face_dataScale = scale.transform(images_array) # 应用规则
# 建立一个聚类为8的点
kmeans = KMeans(n_clusters=8,random_state=12).fit(face_dataScale) # 构建并训练模型


'''  聚类结果可视化  '''
print("waiting for the tsne")
tsne = TSNE(n_components=2,init='random',random_state=17).fit_transform(face_dataScale)    # 使用TSNE进行数据降维，降成两维
df = pd.DataFrame(tsne.embedding_)                    # 将原始数据转换为DataFrame
print("tsne is ok...")
df['labels'] = kmeans.labels_     # 将聚类结果存储进df数据表中
df1 = df[df['labels']==0]
df2 = df[df['labels']==1]
df3 = df[df['labels']==2]
df4=df[df['labels']==3]
df5=df[df['labels']==4]
df6 = df[df['labels']==5]
df7 = df[df['labels']==6]
df8=df[df['labels']==7]
# fig = plt.figure(figsize=(9,6))    # 绘制图形  设定空白画布，并制定大小
plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD',df4[0],df4[1],'yo',df5[0],df5[1],'gray',df6[0],df6[1],'green',df7[0],df7[1],'blue',df8[0],df8[1],'r')
plt.show()                          # 显示图片