__author__ = "Luke Liu"
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
        self.enc_cell=tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
             for _ in range(num_layers)]
        )

        self.dec_cell=tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
             for _ in range(num_layers)]
        )

        # 为 source 和 target分别定义 词向量空间

        # shape convergence is VOCAB_SIZE to HIDDEN_SIZE
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
#定义前反馈的计算流程,计算每一个token的损失与训练后的结果
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

        # 进行编码器的encoder的计算,tf.nn.dynamic_rnn处理不同的长度序列
        with tf.variable_scope("encoder"):
            enc_outputs,enc_state=tf.nn.dynamic_rnn(
                self.enc_cell,src_emb,src_size,dtype=tf.float32
            )
        #enc_outputs的输出应该是（batch_size,max_len,hidden_size)


        # 进行decoder的计算，同样使用时tf.nn.dynamic_rnn
        #将enc_state给decoder作为一个初始的h
        with tf.variable_scope("decoder"):
            dec_outputs,_=tf.nn.dynamic_rnn(
                self.dec_cell,trg_emb,trg_size,initial_state=enc_state
            )
        # decode_outputs的输出应该是（batch_size,max_len,hidden_size)
        '''
        loss evaluation step
        '''
        #计算log_perplexity,与模型代码相同
        #相当于一个flatten操作
        #output shape is (batch_size* trg_size,hidden_size)
        output =tf.reshape(dec_outputs,[-1,hidden_size])
        logits=tf.matmul(output,self.softmax_weight)+self.softmax_bias
        shape_of_logits=tf.shape(logits)
        #计算loss,真个batch的loss
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label,[-1]),logits=logits
        )

        #将填充为0的部分的权重设计为0
        label_weight=tf.sequence_mask(trg_size,maxlen=tf.shape(trg_label)[1],
                                      dtype=tf.float32)
        label_weight=tf.reshape(label_weight,[-1])
        cost=tf.reduce_sum(loss*label_weight)

        cost_per_token=cost/tf.reduce_sum(label_weight)

        trainable_variables = tf.trainable_variables()

        #控制梯度

        grads = tf.gradients(cost/tf.to_float(batch_size),trainable_variables)

        grads,_=tf.clip_by_global_norm(grads,max_grad_norm)

        optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op=optimizer.apply_gradients(zip(grads,trainable_variables))

        return cost_per_token,train_op

#跑一个epoches
def run_epoches(session,cost_op,train_op,saver,step):
    #便利了一次Dataset
    while True:
        try:
            cost,_ = session.run([cost_op,train_op])
            if step%10==0:
                print("After {} steps, per token loss is {}".format(step,cost))

            if step%200==0:
                saver.save(session,checkpoint_path,global_step=step)
            step+=1

        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    initializer=tf.random_uniform_initializer(-0.05,0.05)

    #定义训练的模型：
    with tf.variable_scope("nmt_model",reuse=None,
                           initializer=initializer):
        train_model = NMTModel()

    #定义数据
    data=MakeSrcTrgDataset(src_training_data,trg_training_data,batch_size)
    iterator=data.make_initializable_iterator()
    (src,src_size),(trg_input,trg_label,trg_size)=iterator.get_next()

    #进行forward计算
    cost_op,train_op=train_model.forward(src,src_size,trg_input,trg_label,trg_size)

    #进行训练
    saver=tf.train.Saver()
    step=0
    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(num_epochs):
            print("Epochs {}".format(i+1))
            sess.run(iterator.initializer)
            steps=run_epoches(sess,cost_op,train_op,saver,step)


if __name__ == '__main__':
    main()


