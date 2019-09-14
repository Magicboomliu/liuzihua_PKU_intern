__author__ = "Luke Liu"
#encoding="utf-8"
# -*- coding: utf-8 -*-

# 引入第三方库
import tensorflow as tf
import numpy as np
import urllib
import tarfile
import os
import matplotlib.pyplot as plt
from imageio import imread, imsave, mimsave
from scipy.misc import imresize
import glob

"""
准备工作
"""
# database's path
filename = 'D:/BaiduYunDownload/python_exe/dataset/lfw.tgz'
directory = 'lfw_imgs'
new_dir = 'lfw_new_imgs'
# # 解压文件
# tar = tarfile.open(filename, 'r:gz')
# tar.extractall(path=directory)
# tar.close()
# #统计照片的个数，并且把照片存入new_dir
# count = 0
# for dir_, _, files in os.walk(directory):
#     for file_ in files:
#         img = imread(os.path.join(dir_, file_))
#         imsave(os.path.join(new_dir, '%d.png' % count), img)
#         count += 1
# print(count)
#指定dataset
dataset = 'D:\BaiduYunDownload\python_exe\dataset\scut_faces\AF' # LFW
# dataset = 'celeba' # CelebA
images = glob.glob(os.path.join(dataset, '*.*'))
print(len(images))
# 定义一个输出sample的file
OUTPUT_DIR = 'samples_'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

"""
建立参数

"""
#
batch_size = 100
# noise dim is 100
z_dim = 100
WIDTH = 64
HEIGHT = 64

# 指定输入与是否训练
X = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

#先用sigmoid处理到0-1，然后进过cross-entropy
def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

# 判别器部分, 2 return ,
def discriminator(image, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        #Conv1_: 64 filter is 5, stride is 2, activation is Lekrelu
        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))
        # Conv2_: 128 filter is 5 ,stride is 2,activation is lkeRelu
        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        # BN
        h1 = lrelu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))
        #

        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = lrelu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = lrelu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
       # 展开，但不要连全连接层
        h4 = tf.contrib.layers.flatten(h3)
        # 直接交sigmoid
        h4 = tf.layers.dense(h4, units=1)
        return tf.nn.sigmoid(h4), h4


def generator(z, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
# d is the deep of the image
        d = 4
        h0 = tf.layers.dense(z, units=d * d * 512)
# 增加一个维度
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
# 前面的卷积转置都不加激活函数
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))

# 最后一个卷积装置的输出要经过tanh激活函数
        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh,
                                        name='g')
        return h4

# 定义损失函数
# 产生噪声
g = generator(noise)
d_real, d_real_logits = discriminator(X)
d_fake, d_fake_logits = discriminator(g, reuse=True)
# generator的可训练参数以及discriminator的可训练参数
vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
# generator 的loss以及 discriminator 的loss
loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real)))
loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake)))
loss_g = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake)))
loss_d = loss_d_real + loss_d_fake

# 优化参数
# 关于在batch_norm中，即为更新mean和variance的操作，因此需要update_ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
# 首先优化dis
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
# 然后优化gen
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)


def read_image(path, height, width):
    image = imread(path)
    h = image.shape[0]
    w = image.shape[1]

    if h > w:
        image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
    else:
        image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]

    image = imresize(image, (height, width))
    return image / 255.
# 合成多张图片
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

# 进行训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
samples = []
loss = {'d': [], 'g': []}

saver = tf.train.Saver()
offset = 0
for i in range(60000):
    n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    offset = (offset + batch_size) % len(images)
    batch = np.array([read_image(img, HEIGHT, WIDTH) for img in images[offset: offset + batch_size]])
    batch = (batch - 0.5) * 2
    d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch, noise: n, is_training: True})
    loss['d'].append(d_ls)
    loss['g'].append(g_ls)
     #每优化一次discrimintor,优化两次generator
    sess.run(optimizer_d, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
    print("now is iteratio {}".format(i))

    if i % 50 == 0:
        print("iteration {}, the d_ls is {}, and the g_ls is {}".format(i,d_ls,g_ls))
        #生成图片
        gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})
        gen_imgs = (gen_imgs + 1) / 2
        imgs = [img[:, :, :] for img in gen_imgs]
        gen_imgs = montage(imgs)
        imsave(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i), gen_imgs)
        samples.append(gen_imgs)

plt.plot(loss['d'], label='Discriminator')
plt.plot(loss['g'], label='Generator')
plt.legend(loc='upper right')
plt.savefig(os.path.join(OUTPUT_DIR, 'Loss.png'))
plt.show()
mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=10)
# save the data checkpoint
saver.save(sess, os.path.join(OUTPUT_DIR, 'dcgan_' + dataset), global_step=60000)
