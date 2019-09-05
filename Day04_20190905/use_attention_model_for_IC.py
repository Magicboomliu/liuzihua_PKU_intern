__author__ = "Luke Liu"
#encoding="utf-8"
# First input following modules

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from imageio import imread
import scipy.io
import cv2
import os
import json
# tqdm是python的一个进度条的库，用来显示进度长的长度
from tqdm import tqdm
import pickle

# 规定相关的参数
# 其中maxlen是规定每一个image caption的长度，超过20的把它长度缩小到20，方便进行LSTM处理
batch_size = 128
maxlen = 20
image_size = 224

#VGG19通道均值
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))
'''


Step1 加载一些图片，对应的描述信息以

'''
# 读取数据，1.输入图片的文件夹 2.注解的文件夹
# 返回的数值有id list以及 描述 信息的list,以及id 与 image的array信息组成的字典
def load_data(image_dir, annotation_path):
# 读取注解的信息
    with open(annotation_path, 'r') as fr:
        # 读取注解的json文件
        annotation = json.load(fr)
# 要标记图片的id以对应图片的描述，将id与对应的描述写入一个字典中
    ids = []
    captions = []
    image_dict = {}
# 可以使用进度掉来显示
    for i in tqdm(range(len(annotation['annotations']))):
        # 获得一个注解的信息
        item = annotation['annotations'][i]
        # 获得注解中的描述信息，将所有的小写，而且去除换行信心
        caption = item['caption'].strip().lower()
        #将所有的标点以及特殊的符号换成一个空格
        caption = caption.replace('.', '').replace(',', '').replace("'", '').replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(')', '').replace('-', ' ').split()
        # 将caption中的单词写入一个列表中，如果这个单词大于0
        #放置一个空格进去
        caption = [w for w in caption if len(w) > 0]

#如果这个caption 的长度小于20，保留这个图片与描述信息，写入列表
        if len(caption) <= maxlen:
            #而且这张图片image_id若如果没有读取过的话，
            if not item['image_id'] in image_dict:
                #读取这个信息，array的iamge信息
                img = imread(image_dir + '%012d.jpg' % item['image_id'])
                #获得图片的大小
                h = img.shape[0]
                w = img.shape[1]
                #将图片转化成正方形，保留最主要的部分，不用插值法
                if h > w:
                    img = img[h // 2 - w // 2: h // 2 + w // 2, :]
                else:
                    img = img[:, w // 2 - h // 2: w // 2 + h // 2]
                # 然后将图片转化成规定的大小的正方形
                img = cv2.resize(img, (image_size, image_size))
# 不排除存在黑吧的图片，为其增加一个维度
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
# 然后将其的channel变为3
                    img = np.concatenate([img, img, img], axis=-1)
#将处理后的图片放入image_dict中，一个item[id]对应一个image array
                image_dict[item['image_id']] = img
#然后在id中添加id信息
            ids.append(item['image_id'])
        #在caption中加入caption信息
            captions.append(caption)

    return ids, captions, image_dict

# training文件的jason文件，这个json文件中有annotation的信息
train_json = 'data/train/captions_train2014.json'
train_ids, train_captions, train_dict = load_data('data/train/images/COCO_train2014_', train_json)

# 看一下满足条件的（描述文字<20)的图片序号
print(len(train_ids))

# 这段代码主要来查看一下id对应的一些图片以及相应的caption信息（选择执行）
# data_index = np.arange(len(train_ids))
# np.random.shuffle(data_index)
# N = 4
# data_index = data_index[:N]
# plt.figure(figsize=(12, 20))
# for i in range(N):
#     caption = train_captions[data_index[i]]
#     img = train_dict[train_ids[data_index[i]]]
#     plt.subplot(4, 1, i + 1)
#     plt.imshow(img)
#     plt.title(' '.join(caption))
#     plt.axis('off')


'''

Step2 建立一个词汇对照表，词汇到id,id到词汇
'''
# 建立一个词汇的字典
vocabulary = {}
#对每一个caption中可能出现的单词频率（次数）变成对应编号
for caption in train_captions:
    for word in caption:
        vocabulary[word] = vocabulary.get(word, 0) + 1

#将这个词汇字典点进行排序，按照从大到小的顺序进行排列
vocabulary = sorted(vocabulary.items(), key=lambda x:-x[1])
# 获得对应的词汇表（从大到小）
vocabulary = [w[0] for w in vocabulary]
# 定义一些特殊的符号
word2id = {'<pad>': 0, '<start>': 1, '<end>': 2}
# 把刚才的一些词汇信息加入到word2id字典中去，从标号3开始
#这样word2id前3个是特殊的词汇，后面开始就是词汇表
for i, w in enumerate(vocabulary):
    word2id[w] = i + 3
#将字典变成数字索引在前，而文字信息在后面（先出现的频率高）
id2word = {i: w for w, i in word2id.items()}

# 打印目前词汇表达大小，打印前20个高频词汇，（this is for test!)
print(len(vocabulary), vocabulary[:20])

# 报词汇表，word2id以及id2word变成pickle文件储存起来
with open('dictionary.pkl', 'wb') as fw:
    pickle.dump([vocabulary, word2id, id2word], fw)

# 这可以给定的一个id列表转换为文字
def translate(ids):
    words = [id2word[i] for i in ids if i >= 3]
    return ' '.join(words) + '.'

#这个将描述转换为对应的id，返回一个一个[idex_of_the_caption,captions_id_reflection]的矩阵
def convert_captions(data):
    result = []
    # 在描述开始与描述结束分别加入特殊符号<start>  <end>
    for caption in data:
        # vector is list
        vector = [word2id['<start>']]
        for word in caption:
            if word in word2id:
                vector.append(word2id[word])
        vector.append(word2id['<end>'])
        result.append(vector)
# result最后是所有caption的一个数值对应的list
    #时间很长，我们建立一个进度长来看转化的进度
    #如果不到22就补0，0其实就是<pad>
    array = np.zeros((len(data), maxlen + 2), np.int32)
    for i in tqdm(range(len(result))):
        array[i, :len(result[i])] = result[i]
#将最后的结果转化为一个[idex_of_the_caption,captions_id_reflection]的矩阵
    return array

#执行这个函数，把描述转化为id信息
train_captions = convert_captions(train_captions)
# show some
print("the shape of training captions is :",train_captions.shape)
print("show the first coded captions",train_captions[0])
print("if you do not know what these codes are,dont"
      "not worry,here is the translation"
      ": ",translate(train_captions[0]))


'''
加载模型，首先使用vgg19进行特征提取

'''
#加载模型的参数矩阵
VGG_MODEL = "D:/BaiduYunDownload/python_exe/models/convs/imagenet-vgg-verydeep-19.mat"
vgg = scipy.io.loadmat(VGG_MODEL)
vgg_layers = vgg['layers']

def vgg_endpoints(inputs, reuse=None):
    with tf.variable_scope('endpoints', reuse=reuse):
# 加载权重与偏置
        def _weights(layer, expected_layer_name):
            W = vgg_layers[0][layer][0][0][0][0][0]
            b = vgg_layers[0][layer][0][0][0][0][1]
            layer_name = vgg_layers[0][layer][0][0][3][0]
            assert layer_name == expected_layer_name
            return W, b
#定义卷积层，之后的是relu
        def _conv2d_relu(prev_layer, layer, layer_name):
            W, b = _weights(layer, layer_name)
            W = tf.constant(W)
            b = tf.constant(np.reshape(b, (b.size)))
            return tf.nn.relu(tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b)
#定义平均池化层
        def _avgpool(prev_layer):
            return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        graph = {}
        graph['conv1_1']  = _conv2d_relu(inputs, 0, 'conv1_1')
        graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = _avgpool(graph['conv1_2'])
        graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = _avgpool(graph['conv2_2'])
        graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = _avgpool(graph['conv3_4'])
        graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = _avgpool(graph['conv4_4'])
        graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = _avgpool(graph['conv5_4'])

        return graph

# 输入一张图片
X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
# 输出最后的卷积层的第5个卷积模块第三的卷积层
encoded = vgg_endpoints(X - MEAN_VALUES)['conv5_3']
#我们看一下encode的信息
print(encoded)

'''下面定义lstm部分
'''

k_initializer = tf.contrib.layers.xavier_initializer()
b_initializer = tf.constant_initializer(0.0)
e_initializer = tf.random_uniform_initializer(-1.0, 1.0)

#定义dense层
def dense(inputs, units, activation=tf.nn.tanh, use_bias=True, name=None):
    return tf.layers.dense(inputs, units, activation, use_bias,
                           kernel_initializer=k_initializer, bias_initializer=b_initializer, name=name)

#定义BN层
def batch_norm(inputs, name):
    return tf.contrib.layers.batch_norm(inputs, decay=0.95, center=True, scale=True, is_training=True,
                                        updates_collections=None, scope=name)

#定义dropout层
def dropout(inputs):
    return tf.layers.dropout(inputs, rate=0.5, training=True)


num_block = 14 * 14
num_filter = 512
hidden_size = 1024
embedding_size = 512

#之前应该是（instances,14,14,512)
# 首先将encode reshape成（instances,14*14,512)

encoded = tf.reshape(encoded, [-1, num_block, num_filter])  # batch_size, num_block, num_filter
# 正则化处理
contexts = batch_norm(encoded, 'contexts')
#此时context的shape 也是  # （batch_size, num_block, num_filter）
#输出的结果最长是22（加上了<start>和<end>)
Y = tf.placeholder(tf.int32, [None, maxlen + 2])
#
Y_in = Y[:, :-1]#前面21个
Y_out = Y[:, 1:]#从第2个到第22个
#返回一个布尔类型的张量，然后被转化为float类型，维度与Y_out一样
#word2id['<pad>’]的值是0
mask = tf.to_float(tf.not_equal(Y_out, word2id['<pad>']))

with tf.variable_scope('initialize'):
    #计算均值
    #消失了num_block维度，变成了所有num_block在filter维度上的均值
    context_mean = tf.reduce_mean(contexts, 1)
    #定义最早的状态是1024维度
    state = dense(context_mean, hidden_size, name='initial_state')
    #最早的记忆也是1024维度
    memory = dense(context_mean, hidden_size, name='initial_memory')

#词嵌入，把所有词汇表中词汇嵌入到512维张量
with tf.variable_scope('embedding'):
    embeddings = tf.get_variable('weights', [len(word2id), embedding_size], initializer=e_initializer)
    # 使用tf.nn.embedding_lookup可以读取词向量
    embedded = tf.nn.embedding_lookup(embeddings, Y_in)

with tf.variable_scope('projected'):
    projected_contexts = tf.reshape(contexts, [-1, num_filter])  # batch_size * num_block, num_filter
    #特征映射，注意这一步要以num_filter为分割，每个filer是一个feature vectors
    #讲过一个dense层，所有batch中的特征累加
    projected_contexts = dense(projected_contexts, num_filter, activation=None, use_bias=False,
                               name='projected_contexts')
#将其变化为batch_size, num_block, num_filter的形式
    projected_contexts = tf.reshape(projected_contexts,
                                    [-1, num_block, num_filter])  # batch_size, num_block, num_filter

# 首先建立一个lstm单元
lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
loss = 0
alphas = []

'''
按照次序进行词语生成
'''
for t in range(maxlen + 1):
    with tf.variable_scope('attend'):
        #注意力模块
        h0 = dense(state, num_filter, activation=None, name='fc_state')  # batch_size, num_filter
        h0 = tf.nn.relu(projected_contexts + tf.expand_dims(h0, 1))  # batch_size, num_block, num_filter
        h0 = tf.reshape(h0, [-1, num_filter])  # batch_size * num_block, num_filter
        h0 = dense(h0, 1, activation=None, use_bias=False, name='fc_attention')  # batch_size * num_block, 1
        h0 = tf.reshape(h0, [-1, num_block])  # batch_size, num_block

        alpha = tf.nn.softmax(h0)  # batch_size, num_block
        # contexts:                 batch_size, num_block, num_filter
        # tf.expand_dims(alpha, 2): batch_size, num_block, 1
        context = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), 1, name='context')  # batch_size, num_filter
        alphas.append(alpha)
#选择器
    with tf.variable_scope('selector'):
        beta = dense(state, 1, activation=tf.nn.sigmoid, name='fc_beta')  # batch_size, 1
        context = tf.multiply(beta, context, name='selected_context')  # batch_size, num_filter

    with tf.variable_scope('lstm'):
        h0 = tf.concat([embedded[:, t, :], context], 1)  # batch_size, embedding_size + num_filter
        _, (memory, state) = lstm(inputs=h0, state=[memory, state])

#解码lstm
    with tf.variable_scope('decode'):
        h0 = dropout(state)
        h0 = dense(h0, embedding_size, activation=None, name='fc_logits_state')
        h0 += dense(context, embedding_size, activation=None, use_bias=False, name='fc_logits_context')
        h0 += embedded[:, t, :]
        h0 = tf.nn.tanh(h0)
        h0 = dropout(h0)
        #生成一个概率模型
        logits = dense(h0, len(word2id), activation=None, name='fc_logits')

    loss += tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_out[:, t], logits=logits) * mask[:, t])

    tf.get_variable_scope().reuse_variables()

# 构造优化器，在损失函数中加入注意力正则项，定义优化器

alphas = tf.transpose(tf.stack(alphas), (1, 0, 2)) # batch_size, maxlen + 1, num_block
alphas = tf.reduce_sum(alphas, 1) # batch_size, num_block
attention_loss = tf.reduce_sum(((maxlen + 1) / num_block - alphas) ** 2)
total_loss = (loss + attention_loss) / batch_size

with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
    global_step = tf.Variable(0, trainable=False)
    vars_t = [var for var in tf.trainable_variables() if not var.name.startswith('endpoints')]
    train_op = tf.contrib.layers.optimize_loss(total_loss, global_step, 0.001, 'Adam', clip_gradients=5.0, variables=vars_t)

'''
train the model
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
OUTPUT_DIR = 'model'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
# 使用了tensorboard
tf.summary.scalar('losses/loss', loss)
tf.summary.scalar('losses/attention_loss', attention_loss)
tf.summary.scalar('losses/total_loss', total_loss)
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(OUTPUT_DIR)

epochs = 20
#一共训练20论
for e in range(epochs):
    train_ids, train_captions = shuffle(train_ids, train_captions)

    for i in tqdm(range(len(train_ids) // batch_size)):
        #定义batch的大小
        X_batch = np.array([train_dict[x] for x in train_ids[i * batch_size: i * batch_size + batch_size]])
        Y_batch = train_captions[i * batch_size: i * batch_size + batch_size]
        _ = sess.run(train_op, feed_dict={X: X_batch, Y: Y_batch})
        if i > 0 and i % 100 == 0:
        #每100记录一下
            writer.add_summary(sess.run(summary,
                                        feed_dict={X: X_batch, Y: Y_batch}),
                               e * len(train_ids) // batch_size + i)
            writer.flush()
#最后储存
    saver.save(sess, os.path.join(OUTPUT_DIR, 'image_caption'))

#question how to show