
#encoding="utf-8"

import tensorflow as tf
import os
from data_batching_process import MakeSrcTrgDataset
src_training_data  =  'preprocessed_data/train.en' # source file
trg_training_data  = 'preprocessed_data/train.zh' # target file
checkpoint_path='./path/to/seq2seq_ckpt'
if not os.path.exists(checkpoint_path):
    #创建多级目录
    os.makedirs(checkpoint_path)

# 一些构建LSTM的decoder与encoder需要用到的参数
hidden_size=1024  #lstm 的隐藏层规模
num_layers = 2  #使用2层lstm
src_vocab_size = 10000 #源词汇表的大小
trg_vocab_size = 4000  #目标词汇表的大小

batch_size = 100
num_epochs = 5
keep_pro = 0.8
# 这个用于控制梯度膨胀
max_grad_norm = 5
# embedding层与softmax层参数共享（transpose一下就可以了）
share_emb_and_softmax=True

#定义NMT(Neural Network Machine Translation)


class NMTModel(object):
    def __init__(self):
        # 首先定义最基本的encoder与decoder的 Lstm 结构
        self.enc_cell_fw=tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        self.enc_cell_bw=tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        # 为 source 和 target分别定义 词向量空间
        # shape convergence is VOCAB_SIZE to HIDDEN_SIZE

        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
             for _ in range(num_layers)]
        )


        self.src_embedding=tf.get_variable(
            "src_emb",[src_vocab_size,hidden_size]
        )
        self.trg_embedding=tf.get_variable(
            "trg_emb",[trg_vocab_size,hidden_size]
        )

        #解码器需要用的softmax层
        #定义softmax层的变量
        if share_emb_and_softmax:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight=tf.get_variable("weight",
                    [hidden_size,trg_vocab_size])
        self.softmax_bias = tf.get_variable(
            "softmax_bias",[trg_vocab_size]
        )

    def forward(self,src_input,src_size,trg_input,trg_label,trg_size):
        #src_input的shape 应该是（batch_size,src_size)
        batch_size = tf.shape(src_input)[0]
        #将输入与输出转化为词向量（embedding层输出）
        src_emb=tf.nn.embedding_lookup(self.src_embedding,src_input)
        trg_emb=tf.nn.embedding_lookup(self.trg_embedding,trg_input)
        #进行dropout操作
        src_emb=tf.nn.dropout(src_emb,keep_pro)
        trg_emb=tf.nn.dropout(trg_emb,keep_pro)
        # trg_emb的shape应该是（batch_size,max_len,embedding_layer)

        with tf.variable_scope('encoder'):
            # 构造编码器时，使用birdirectional_dynamic_rnn构造双向循环网络。
            # 双向循环网络的顶层输出enc_outputs是一个包含两个张量的tuple，每个张量的
            # 维度都是[batch_size, max_time, HIDDEN_SIZE],代表两个LSTM在每一步的输出
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw, self.enc_cell_bw, src_emb,
                                                                     src_size, dtype=tf.float32)
            # 将两个LSTM输出拼接为一个张量
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        # 使用dynamic_rnn构造解码器
        with tf.variable_scope('decoder'):
            # 选择注意力权重的计算模型。BahdanauAttention是使用一个隐藏层的前馈神经网络
            # memory_sequence_length是一个维度为[batch_size]的张量，代表batch中每个句子的长度
            # Attention需要根据这个信息把填充位置的注意里权重设置为0
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hidden_size, enc_outputs,
                                                                       memory_sequence_length=src_size)
            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism,
                                                                 attention_layer_size=hidden_size)
            # 使用attention_cell和dynamic_rnn构造编码器
            # 这里没有指定init_state,也就是没有使用编码器的输出来初始化输入，而完全依赖注意力作为信息来源
            dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell, trg_emb, trg_size, dtype=tf.float32)

            # decode_outputs的输出应该是（batch_size,max_len,hidden_size)
        '''
        loss evaluation step
        '''
        # 计算log_perplexity,与模型代码相同
        # 相当于一个flatten操作
        # output shape is (batch_size* trg_size,hidden_size)
        output = tf.reshape(dec_outputs, [-1, hidden_size])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        shape_of_logits = tf.shape(logits)
        # 计算loss,真个batch的loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits
        )

        # 将填充为0的部分的权重设计为0
        label_weight = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1],
                                        dtype=tf.float32)
        label_weight = tf.reshape(label_weight, [-1])
        cost = tf.reduce_sum(loss * label_weight)

        cost_per_token = cost / tf.reduce_sum(label_weight)

        trainable_variables = tf.trainable_variables()

        # 控制梯度

        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)

        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        return cost_per_token, train_op

# 跑一个epoches
def run_epoches(session, cost_op, train_op, saver, step):
    # 便利了一次Dataset
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After {} steps, per token loss is {}".format(step, cost))

            if step % 200 == 0:
                saver.save(session, checkpoint_path, global_step=step)
            step += 1

        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练的模型：
    with tf.variable_scope("nmt_model", reuse=None,
                           initializer=initializer):
        train_model = NMTModel()

    # 定义数据
    data = MakeSrcTrgDataset(src_training_data, trg_training_data, batch_size)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 进行forward计算
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    # 进行训练
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(num_epochs):
            print("Epochs {}".format(i + 1))
            sess.run(iterator.initializer)
            steps = run_epoches(sess, cost_op, train_op, saver, step)

if __name__ == '__main__':
    main()





