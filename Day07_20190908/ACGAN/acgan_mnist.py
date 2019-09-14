__author__ = "Luke Liu"
#encoding="utf-8"

#import reference modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, imageio
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
checkpoint="D:/BaiduYunDownload/python_exe/dataset/path/to_cgan/acgan.ckpt"
''' 布置网络的相关参数 '''

batch_size = 100
z_dim = 100
WIDTH = 28
HEIGHT = 28
#  共有 0-9 10个类别
LABEL = 10
#定义output文件夹
OUTPUT_DIR = 'samples'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

#定义输入的格式

X = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,1],name='input-x')

noise= tf.placeholder(dtype=tf.float32,shape=[None,z_dim],name='noise')
y_label_noise=tf.placeholder(dtype=tf.float32,shape=[None,LABEL],name='C_vector')

#是否训练，验证的时候，is_training 设置为 False
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

# 判别器使用的lrelu函数
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

# 使用DCGAN，所以使用
def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

# 判别器训练，返回2个值，1个是否是真实的，一个属于哪个分类
def discriminator(image, reuse=None, is_training=is_training):
    #这次没有条件输入，单数加入了类别的判定
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))

        h1 = lrelu(tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same'))

        h2 = lrelu(tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same'))

        h3 = lrelu(tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same'))

        h4 = tf.contrib.layers.flatten(h3)
        #最后输出2个值，Y_是分类出的logits，分类出是真实的分数
        Y_ = tf.layers.dense(h4, units=LABEL)
        h4 = tf.layers.dense(h4, units=1)
        return h4, Y_

def generator(z, label, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 3
        z = tf.concat([z, label], axis=1,name='vector_input')
        # 此时z的shape为（None,110)
        h0 = tf.layers.dense(z, units=d * d * 512)

        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))

        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=1, strides=1, padding='valid',
                                        activation=tf.nn.tanh,
                                        name='g')
        return h4

#定义loss
g = generator(noise, y_label_noise)

d_cls_logits,d_score_logits = discriminator(X)
d_fake_cls_logits,d_fake_score_logits=discriminator(g,reuse=True)

#这是判别器的损失
loss_d_cls=tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.argmax(y_label_noise, 1),logits=d_cls_logits)
loss_d_cls=tf.reduce_mean(loss_d_cls)


loss_d_score_1= sigmoid_cross_entropy_with_logits(d_score_logits,tf.ones_like(d_score_logits))
loss_d_score_2=sigmoid_cross_entropy_with_logits(d_fake_score_logits,tf.zeros_like(d_fake_score_logits))
loss_d_score_1=tf.reduce_mean(loss_d_score_1)
loss_d_score_2=tf.reduce_mean(loss_d_score_2)
loss_d_dis=loss_d_score_1+loss_d_score_2

loss_d_total=loss_d_cls+loss_d_dis

#这是生成器的损失

loss_g_dis = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_score_logits,tf.ones_like(d_fake_score_logits)))
loss_g_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(

labels=tf.argmax(y_label_noise, 1),logits=d_fake_score_logits
))
loss_g_total=loss_g_cls+loss_g_dis

vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

#定义优化器
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d_total, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g_total, var_list=vars_g)

def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


with tf.Session() as sess:
    #全局变量初始化
    init_op=tf.global_variables_initializer()
    sess.run([init_op])


    #加载数据数据，主要做测试用
    # this is generator's data
    z_samples=np.random.uniform(-1.0,1.0,[batch_size,z_dim]).astype('float32')
    y_sample_labels=np.zeros([batch_size,LABEL])
    for i in range(LABEL):
        for j in range(LABEL):
            y_sample_labels[i*LABEL+j,i]=1
    samples = []
    loss = {'d': [], 'g': []}
    saver=tf.train.Saver()
    for i in tqdm(range(60000)):
        #每次随机取noise
        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
        #加载真实的数据，并做整理
        batch, label = mnist.train.next_batch(batch_size=batch_size)
        batch = np.reshape(batch, [batch_size, HEIGHT, WIDTH, 1])
        batch = (batch - 0.5) * 2
        #注意生成器使用的label与判别器使用的label 一样
        yn = np.copy(label)
        d_ls, g_ls = sess.run([loss_d_total, loss_g_total],
                              feed_dict={X: batch, noise: n, y_label_noise: yn, is_training: True})

        loss['d'].append(d_ls)
        loss['g'].append(g_ls)

        sess.run(optimizer_d, feed_dict={X: batch, noise: n, y_label_noise: yn, is_training: True})
        sess.run(optimizer_g, feed_dict={X: batch, noise: n, y_label_noise: yn, is_training: True})
        sess.run(optimizer_g, feed_dict={X: batch, noise: n, y_label_noise: yn, is_training: True})

        if i % 100 == 0:
            print(i, d_ls, g_ls)
            print("have fininshed {} steps".format(i))
            gen_imgs = sess.run(g, feed_dict={noise: z_samples, y_label_noise:y_sample_labels, is_training: False})
            saver.save(sess,checkpoint,global_step=i)
            gen_imgs = (gen_imgs + 1) / 2

            imgs = [img[:, :, 0] for img in gen_imgs]
            gen_imgs = montage(imgs)
            imageio.imsave(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i), gen_imgs)
            samples.append(gen_imgs)

    plt.plot(loss['d'], label='Discriminator')
    plt.plot(loss['g'], label='Generator')
    plt.legend(loc='upper right')
    plt.savefig('Loss.png')
    plt.show()
    imageio.mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=5)







