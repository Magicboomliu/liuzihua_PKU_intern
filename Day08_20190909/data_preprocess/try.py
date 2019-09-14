__author__ = "Luke Liu"
# encoding="utf-8"
import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph("classifier1/sp.ckpt.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./classifier"))
    graph = tf.get_default_graph()
    input_x = graph.get_tensor_by_name("input_image:0")
    Y_ = graph.get_tensor_by_name('scores:0')
    # 加载你想要进行的操作
    result = graph.get_tensor_by_name("vgg/fc8/relu_layer:0")

