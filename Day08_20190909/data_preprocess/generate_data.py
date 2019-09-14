__author__ = "Luke Liu"
#encoding="utf-8"
import pickle
import numpy as np
import  os
import  cv2
import  matplotlib.pyplot as plt
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
paths_labels=unpickle("FACES_P_S")
images_path = paths_labels['images_path']
labels=paths_labels['scores_path']
#scores rank 预处理
labels=[round(label,2) for label in labels]
index=[0,1,2,3,4,5,6,7,8]
score_rank=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
index_scores_dict=dict(list(zip(score_rank,index)))
redefined_labels=[]
redefined_labels_index=[]
for label in labels:
    a=[]
    for i in range(len(score_rank)):
        a.append(abs(label-score_rank[i]))
    label=score_rank[np.argmin(a)]
    redefined_labels.append(label)
for i in redefined_labels:
    redefined_labels_index.append(index_scores_dict[i])



images_array=np.zeros((5500,299,299,3),dtype=np.float32)
for i in range(5500):
    img=cv2.imread(images_path[i])
    im=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(299,299))
    images_array[0,:,:,:]+=im
    print("finis {}/5500".format(i))
images_array=images_array/255.0
validation_images = images_array[:100]
validation_scores=redefined_labels_index[:100]

import tensorflow as tf
import os
import  tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

train_file="D:/BaiduYunDownload/python_exe/dataset/scut_faces/path/to/saved"
if not os.path.exists(train_file):
    os.makedirs(train_file)
ckpt_file="D:/BaiduYunDownload/python_exe/models/convs/inception/inception_v3.ckpt"

#定义训练使用的参数
learning_rate = 0.0001
steps = 300
batch = 32
classes = 8
#，要训练的是最后的全连接层
checkpoint_exclude_scopes='InceptionV3/Logits,InceptionV3/AuxLogits'
trainable_scopes='InceptionV3/Logits,InceptionV3/AuxLogits'

#加载所有固定的不需要动的参数
def get_tuned_variables():
    #判断要移除的层
    exclusions = {scope.strip() for scope in checkpoint_exclude_scopes.split(",")}
    variables_to_restore=[]
    for var in slim.get_model_variables():
        excluded=False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded=True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

#获得所有需要训练的var
def get_trainable_variables():
    scopes=[scope.strip() for scope in trainable_scopes.split(",")]
    variables_to_train=[]
    for scope in scopes:
        variables =tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,scope
        )
        variables_to_train.append(variables)
    return variables_to_train

# 加载数据进行操作
# shuffle the data
# state= np.random.get_state()
# np.random.shuffle(images_array)
# np.random.set_state(state)X
# np.random.shuffle(labels)

#定义输入与输出X
X=tf.placeholder(dtype=tf.float32,shape=[None,299,299,3],name="input_image")
scores=tf.placeholder(dtype=tf.float32,shape=[None],name="scores")

with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits,_=inception_v3.inception_v3(
        X,num_classes=classes)

#获得trainable data
trainable_variables=get_trainable_variables()
#使用cross_entropy softmax
'''定义losses'''
tf.losses.softmax_cross_entropy(tf.one_hot(redefined_labels_index,classes),
                                logits,weights=1.0)
'''定义优化'''
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(
    tf.losses.get_total_loss()
)
'''计算正确率'''

with tf.name_scope('evaluation'):
    correct_prediction = tf.equal(tf.argmax(logits,1),redefined_labels_index
                                  )
    evaluation_step=tf.reduce_mean(tf.cast(
        correct_prediction,tf.float32
    ))
# 定义加载模型的参数
load_fn = slim.assign_from_checkpoint_fn(
    ckpt_file,
    get_tuned_variables(),
    ignore_missing_vars=True
)
saver=tf.train.Saver()

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    print("loading tuned variables from {}".format(ckpt_file))
    load_fn(sess)

    #开始训练

    start = 0
    end = batch
    for i in range(steps):
        sess.run(train_step,feed_dict={
            X:images_array[start:end],
            scores:redefined_labels_index[start:end]
        })
        # output log
        if i%30==0 or i+1==steps:
            saver.save(sess,train_file,global_step=i)
            validation_accuracy = sess.run(evaluation_step,feed_dict={
                X:validation_images[start:end],scores:validation_scores[start:end]
            })
            print("Step {},validation accuracy is {}%".format(i,validation_accuracy*100))
        start=end
        if start == 5500:
            start = 0

        end = start+batch
        if end>5500:
            end = 5500

if __name__ == '__main__':
    tf.app.run()





