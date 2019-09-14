__author__ = "Luke Liu"
#encoding="utf-8"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, imageio
from tqdm import tqdm
from PIL import  Image

dataset_path='D:\BaiduYunDownload\python_exe\dataset\scut_faces\Images'
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
images_paths=[os.path.join(dataset_path,i) for i in images_path]
for i in range(2000):
    img=Image.open(images_paths[i])
    img=img.resize((64,64))
    img=np.asarray(img)
    im=img/255.
    images_array[0,:,:,:]+=im
    images_array=(images_array-0.5)*2
    print("finis {}/2000".format(i+1))

#数据处理完毕

#此时的mnist的type是Dataset

batch_size = 100
z_dim = 100
WIDTH = 64
HEIGHT = 64
CHANNEL=3
#  共有 0-9 10个类别
LABEL = 8
#定义output文件夹
OUTPUT_DIR = 'samples'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
# real data  ,因为2者还有合并，所有在这里shape要匹配
X = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='X')
y_label = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, LABEL], name='y_label')

# gen的data
noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
y_noise = tf.placeholder(dtype=tf.float32, shape=[None, LABEL], name='y_noise')

#是否训练，验证的时候，is_training 设置为 False
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

# 判别器使用的lrelu函数
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

# 使用DCGAN，所以使用
def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

#判别器部分
def discriminator(image, label, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        # 在最后一位进行（None，Height,weight，11）
        h0 = tf.concat([image, label], axis=3)

        h0 = lrelu(tf.layers.conv2d(h0, kernel_size=5, filters=64, strides=2, padding='same'))

        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = lrelu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = lrelu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        h3_shape = tf.shape(h3)
        h4 = tf.contrib.layers.flatten(h3)
        h4 = tf.layers.dense(h4, units=1)

        return tf.nn.sigmoid(h4), h4,h3_shape


def generator(z, label, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 4
        z = tf.concat([z, label], axis=1)
        print(z.shape)
        # 此时z的shape为（None,110)
        h0 = tf.layers.dense(z, units=d * d * 512)

        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        print(h0.shape)
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        print(h3.shape)
        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh,
                                        name='g')
        print(h4.shape)
        return h4

g = generator(noise, y_noise)
print(tf.shape(g))
print("??????????????????????????????????")

d_real, d_real_logits,h3_shape = discriminator(X, y_label)
#第二次使用判别器，在一次iteration里面，所以reuse设置成True.
d_fake, d_fake_logits,h3_shapes= discriminator(g, y_label, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
#正真实的损失
loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real_logits)))
loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake_logits)))
loss_g = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake_logits)))

loss_d = loss_d_real + loss_d_fake

#定义一下损失
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)


def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m

#开始训练

with tf.Session() as sess:
    #全局变量初始化
    init_op=tf.global_variables_initializer()
    sess.run([init_op])
    label = tf.one_hot(redefined_labels_index, 8, axis=1)
    label=sess.run(label)
    #加载数据数据，主要做测试用
    # this is generator's data
    z_samples=np.random.uniform(-1.0,1.0,[batch_size,z_dim]).astype('float32')
    y_sample_labels=np.zeros([batch_size,LABEL])
    for i in range(LABEL):
        for j in range(LABEL):
            y_sample_labels[i*LABEL+j,i]=1

    samples = []
    loss = {'d': [], 'g': []}
#开始训练
    for i in tqdm(range(60000)):
        #每次随机取noise
        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
        #加载真实的数据，并做整理
        start = 0
        end = batch_size
        #注意生成器使用的label与判别器使用的label 一样
        yn = np.copy(label[start:end])
        yl = np.reshape(label[start:end], [batch_size, 1, 1, LABEL])
        yl = yl * np.ones([batch_size, HEIGHT, WIDTH, LABEL])

        d_ls, g_ls = sess.run([loss_d, loss_g],
                              feed_dict={X: images_array[start:end], noise: n, y_label: yl, y_noise: yn, is_training: True})

        loss['d'].append(d_ls)
        loss['g'].append(g_ls)

        sess.run(optimizer_d, feed_dict={X: images_array[start:end], noise: n, y_label: yl, y_noise: yn, is_training: True})
        sess.run(optimizer_g, feed_dict={X: images_array[start:end], noise: n, y_label: yl, y_noise: yn, is_training: True})
        sess.run(optimizer_g, feed_dict={X: images_array[start:end], noise: n, y_label: yl, y_noise: yn, is_training: True})
        if i % 100 == 0:
            print(i, d_ls, g_ls)
            print("have fininshed {} steps".format(i))
            gen_imgs = sess.run(g, feed_dict={noise: z_samples, y_noise:y_sample_labels, is_training: False})
            gen_imgs=(gen_imgs+1)/2
            imgs = [img[:, :, 0] for img in gen_imgs]
            gen_imgs = montage(imgs)
            imageio.imsave(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i), gen_imgs)
            samples.append(gen_imgs)

        start = end
        # output log
        if start == 2000:
            start = 0
        end = start + batch_size
        if end > 2000:
            end = 2000

    plt.plot(loss['d'], label='Discriminator')
    plt.plot(loss['g'], label='Generator')
    plt.legend(loc='upper right')
    plt.savefig('Loss.png')
    plt.show()
    imageio.mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=5)
    saver=tf.train.Saver()
    saver.save(sess,save_path=r'D:\BaiduYunDownload\python_exe\dataset\path\123\fcgan.ckpt')


