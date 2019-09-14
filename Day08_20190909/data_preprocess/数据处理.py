__author__ = "Luke Liu"
#encoding="utf-8"
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import os
import  numpy as np
from  PIL import  Image
#加载数据
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
one_hot=tf.one_hot(redefined_labels_index,8,axis=1)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run([init_op])
    a = sess.run([one_hot])
    print(a[:5])