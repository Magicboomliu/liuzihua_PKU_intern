__author__ = "Luke Liu"
#encoding="utf-8"

# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

batch_size = 100
z_dim = 100
dataset = 'lfw_new_imgs'
# dataset = 'celeba'

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

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('samples_' + dataset, 'dcgan_' + dataset + '-60000.meta'))
saver.restore(sess, tf.train.latest_checkpoint('samples_' + dataset))
# 获得当前的图片
graph = tf.get_default_graph()


'''
define the placeholder

'''

# 获得generator的Input
g = graph.get_tensor_by_name('generator/g/Tanh:0')
# 获得noise的值
noise = graph.get_tensor_by_name('noise:0')
# 获得是否train
is_training = graph.get_tensor_by_name('is_training:0')
#

'''
get the data
'''
n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

# 训练得到最后的结果
gen_imgs = sess.run(g, feed_dict={noise: n, is_training: False})

gen_imgs = (gen_imgs + 1) / 2
imgs = [img[:, :, :] for img in gen_imgs]
gen_imgs = montage(imgs)
#将gen_imgs 的值控制在0到1之间。
gen_imgs = np.clip(gen_imgs, 0, 1)

plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(gen_imgs)
plt.show()