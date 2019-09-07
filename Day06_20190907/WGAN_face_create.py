__author__ = "Luke Liu"
#encoding="utf-8"
# continue to use lfw datasets to apply WGAN

'''Import inference modules'''

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from imageio import imread, imsave, mimsave
import cv2
import glob
from tqdm import tqdm

#指定数据库，lfw face dataset，一共有13233个faces
dataset = 'D:\BaiduYunDownload\python_exe\dataset\lfw_new_imgs'
images = glob.glob(os.path.join(dataset, '*.*'))

# 一些网络参数
batch_size = 100
#输入噪声的dimension
z_dim = 100
#  图片的size
WIDTH = 64
HEIGHT = 64
# gp 惩罚项的比重
LAMBDA = 10
DIS_ITERS = 3 # 5

#定义输入的样例图片文件夹
OUTPUT_DIR = 'D:/BaiduYunDownload/python_exe/dataset/wgan/samples_lfw'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
#指定输入的格式

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, HEIGHT, WIDTH, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='noise')
# 是否训练
is_training = tf.placeholder(dtype=tf.bool, name='is_training')
# 定义encoder卷积层使用leRelu函数
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


#定义生成器，判别器部分，注意需要去掉Batch Normalization，否则会导致batch之间的相关性，
# 从而影响gradient penalty的计算
def discriminator(image, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):

        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))

        h1 = lrelu(tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same'))

        h2 = lrelu(tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same'))

        h3 = lrelu(tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same'))
        #h3的shape是（4,4,512）
        h4 = tf.contrib.layers.flatten(h3)
        # h4的shape是（4*4*512）
        #最后输出一个scaler（Batch_Size,)
        h4 = tf.layers.dense(h4, units=1)
        return h4

# 使用了batch normalize
def generator(z, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        #将（None ,100) 变成（4,4,512）
        d = 4
        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))

        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh,
                                    name='g')
        # 此时的h4 是 （64,64,3）
        return h4

'''
定义一波损失函数
'''
#生成的图片g
g = generator(noise)
#判别器FW出的d_real
d_real = discriminator(X)
#判别器FW出的generator产生图片的fage
d_fake = discriminator(g, reuse=True)

#定义一下损失,计算这个batch的总分（取—）
loss_d_real = -tf.reduce_mean(d_real)  #越小越好
loss_d_fake = tf.reduce_mean(d_fake)  #越大越好
loss_g = -tf.reduce_mean(d_fake)  #越小越好
loss_d = loss_d_real + loss_d_fake  # 最大化这一项

#添加一个 1-Lapschitz1 惩罚项 gradient_penalty
#取值在real data 与 generate data之间进行采样
alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
#采样过程
interpolates = alpha * X + (1 - alpha) * g
#计算梯度
grad = tf.gradients(discriminator(interpolates, reuse=True), [interpolates])[0]
#这里使用时L2距离
slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
#得到gp的值
gp = tf.reduce_mean((slop - 1.) ** 2)
#在discriminator的损失中加入gp，使其不会被训练的太好
# this is the total loss
loss_d += LAMBDA * gp

#规定好所有generator与discriminatior的参数
vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

#首先保证正则化，然后使用规定优化的函数，在这里使用adam
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)
#读取图片，并且调整图片的大小
def read_image(path, height, width):
    image = imread(path)
    h = image.shape[0]
    w = image.shape[1]

    if h > w:
        image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
    else:
        image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]

    image = cv2.resize(image, (width, height))
    return image / 255.

# 合并图片，一张图片显示多张人脸
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

# 随机产生批量的数据，主要是real data 的收集
def get_random_batch(nums):
    #抽取一些real data
    img_index = np.arange(len(images))
    np.random.shuffle(img_index)
    img_index = img_index[:nums]
    #加载一个batch的数据作为一个合成的array
    batch = np.array([read_image(images[i], HEIGHT, WIDTH) for i in img_index])
    #这个做正则化处理
    batch = (batch - 0.5) * 2
    return batch


#创建会话，训练数据
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 随即将采样成generator需要输入的数据
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
samples = []
#创建一个loss的字典，存放generator loss 与discriminator loss
loss = {'d': [], 'g': []}

for i in tqdm(range(60000)):
    #训练3次gen之后，训练一次dis
    for j in range(DIS_ITERS):
        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
        batch = get_random_batch(batch_size)
        _, d_ls = sess.run([optimizer_d, loss_d], feed_dict={X: batch, noise: n, is_training: True})

    _, g_ls = sess.run([optimizer_g, loss_g], feed_dict={X: batch, noise: n, is_training: True})

    loss['d'].append(d_ls)
    loss['g'].append(g_ls)
#每500节生成一张图片，并保留

    if i % 50 == 0:
        print(i, d_ls, g_ls)

        gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})
        gen_imgs = (gen_imgs + 1) / 2
        imgs = [img[:, :, :] for img in gen_imgs]
        gen_imgs = montage(imgs)
        plt.axis('off')
        plt.imshow(gen_imgs)
        imsave(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i), gen_imgs)
        plt.show()
        samples.append(gen_imgs)

#画出loss
plt.plot(loss['d'], label='Discriminator')
plt.plot(loss['g'], label='Generator')
plt.legend(loc='upper right')
plt.savefig(os.path.join(OUTPUT_DIR, 'Loss.png'))
plt.show()
mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=10)
#训练结束后，保留最后的模型与参数
saver = tf.train.Saver()
saver.save(sess, os.path.join(OUTPUT_DIR, 'wgan_' + dataset), global_step=60000)