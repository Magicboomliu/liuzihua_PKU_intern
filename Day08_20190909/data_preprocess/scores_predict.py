__author__ = "Luke Liu"

#encoding="utf-8"
# This code use the structure of VGG16 Codes.
import  tensorflow as tf
import pickle
import numpy as np
import  os
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
paths_labels=unpickle("FACES_P_S")
images_path = paths_labels['images_path']
labels=paths_labels['scores_path']
#scores rank 预处理
scores=[round(label,2) for label in labels]

images_array=np.zeros((5500,224,224,3),dtype=np.float32)
for i in range(5500):
    img=Image.open(images_path[i])
    img=img.resize((224,224))
    image=np.asarray(img)
    im=image/255.
    images_array[0,:,:,:]+=im
    print("finis {}/5500".format(i))
print(images_array.shape)

# 输出op的name
def print_activation(t):
    print(t.op.name,'',t.get_shape().as_list())
# input_op 就是输入，name 为这个卷积层命名，kh,kw 代表卷积核的大小。
# dh,dw 代表 步长的高和步长的宽度。 # p 代表参数列表
# 定义conv2d操作的函数
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    #  get channels /dimenons
    n_in=input_op.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        kernel= tf.get_variable("w",
                                shape=[kh,kw,n_in,n_out ],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')

        biases=tf.get_variable("biases",[n_out],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),trainable=True
                               )

        z = tf.nn.bias_add(conv,biases)

        activation = tf.nn.relu(z)

        p+=[kernel,biases]

        print_activation(activation)

        return activation
# 定义full_connected layer 的函数
def fc_op(input_op,name,n_out,p,regularizer):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name)as scope:
        weights = tf.get_variable("W",
                                  shape=[n_in,n_out],dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases=tf.get_variable("biases",shape=[n_out],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        tf.add_to_collection("losses",regularizer(weights))
        activation=tf.nn.relu_layer(input_op,weights,biases)

        p+=[weights,biases]
        print_activation(activation)
        return  activation
# 定义池化层函数
def mpool_op(input_op,name,kh,kw,dh,dw):
    with tf.variable_scope(name) as scope:

        pool= tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],
                              strides=[1,dh,dw,1],
                              padding="SAME",
                              name=name)
        print_activation(pool)
        return pool
# 开始定义VGG16 的网络框架，VGG-16 一共有6个部分组成，
# 其中前5个部分为卷积网络，最后一部分为全连接网络
# 注意 keep_prob是指dropout Rate
def inference_op(input_op,keep_prob,regularizer):
    with tf.variable_scope("vgg"):
        # 参数列表
        p=[]

        # input_image is 224*224*3

        #  首先是第一个卷积部分（2个卷积层和一个最大池化层），3,3 filter 64个, stride=1
        # 卷积层1_1
        conv1_1 = conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
        # 卷积层1_2
        conv1_2=conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
        # 最大池化1
        mp1 = mpool_op(conv1_2,name='maxpooling1',kh=2,kw=2,dh=2,dw=2)

        # 第二个卷积的部分（2个卷积层和一个最大池化层),3 3 filter 128, stride=1

        conv2_1=conv_op(mp1,name='conv2_1',kh=3,kw=3,n_out=128,dw=1,dh=1,p=p)
        conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dw=1, dh=1, p=p)
        mp2=mpool_op(conv2_2,name='maxpooling2',kh=2,kw=2,dh=2,dw=2)

        # 第三个卷积的部分（3个卷积层和一个最大池化层),3 3 filter 256, stride=1
        conv3_1 = conv_op(mp2, name='conv3_1', kh=3, kw=3, n_out=256, dw=1, dh=1, p=p)
        conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dw=1, dh=1, p=p)
        conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dw=1, dh=1, p=p)
        mp3 = mpool_op(conv3_3, name='maxpooling3', kh=3, kw=3, dh=2, dw=2)


        # 第四个卷积的部分（3个卷积层和一个最大池化层），3，3 filter 512,stride=1
        conv4_1 = conv_op(mp3, name='conv4_1', kh=3, kw=3, n_out=512, dw=1, dh=1, p=p)
        conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dw=1, dh=1, p=p)
        conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dw=1, dh=1, p=p)
        mp4 = mpool_op(conv4_3, name='maxpooling4', kh=3, kw=3, dh=2, dw=2)

        # 第5个卷积的部分
        conv5_1 = conv_op(mp4, name='conv5_1', kh=3, kw=3, n_out=512, dw=1, dh=1, p=p)
        conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dw=1, dh=1, p=p)
        conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dw=1, dh=1, p=p)
        mp5 = mpool_op(conv5_3, name='maxpooling5', kh=3, kw=3, dh=2, dw=2)


        # flatten 操作

        shp=mp5.get_shape()
        flattened_shape=shp[1].value *shp[2].value*shp[3].value
        resh1=tf.reshape(mp5,[-1,flattened_shape],name='resh1')

        # 第 6 部分 全连接层(2个全连接层units=4096,最后有一个输出层units is 10000
        fc6 = fc_op(resh1,"fc6",n_out=4096,p=p,regularizer=regularizer)
        fc6_drop = tf.nn.dropout(fc6,keep_prob=keep_prob,name='fc6_drop')

        fc7 = fc_op(fc6_drop,'fc7',n_out=4096,p=p,regularizer=regularizer)
        fc7_drop = tf.nn.dropout(fc7,keep_prob=keep_prob,name='fc7_drop')

        fc8=fc_op(fc7_drop,'fc8',n_out=1,p=p,regularizer=regularizer)
        return fc8,p



def optimizer(loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step= tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss,var_list=vars_g)
    return train_step
steps = 3000
#定义输入与输出
X=tf.placeholder(dtype=tf.float32,shape=[None,224,224,3],name="input_image")
ground_T=tf.placeholder(dtype=tf.float32,shape=[None],name="scores")
keep_prob = tf.placeholder(tf.float32)
Regu_rate = 0.0001
batch_size = 100
regularizer_L2 = tf.contrib.layers.l2_regularizer(Regu_rate)

fc8, p = inference_op(X, keep_prob, regularizer_L2)
tf.losses.mean_squared_error(labels=ground_T,predictions=tf.argmax(fc8,1),weights=1.0)

trainable=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
train_step = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(tf.losses.get_total_loss())


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op, feed_dict={keep_prob: 0.5})
    print("OK")
    start = 0
    end = batch_size
    for i in range(steps):
        sess.run(train_step, feed_dict={
            X: images_array[start:end],
            keep_prob:0.5,
            ground_T: scores[start:end]
        })
        if i % 30 == 0 or i + 1 == steps:
            print("now the loss is :",sess.run([tf.losses.get_total_loss()],feed_dict=
            {
                X: images_array[start:end],
                keep_prob:0.5,
                ground_T: scores[start:end]
            }))
        start = end
        # output log
        if start == 5500:
            start = 0
        end = start + batch_size
        if end > 5500:
            end = 5500


if __name__ == '__main__':
    tf.app.run()




