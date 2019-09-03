__author__ = "Luke Liu"
#encoding="utf-8"
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

batch_size = 100
# 开始层
original_dim = 784
intermediate_dim = 256
# 隐层
latent_dim = 2
epochs = 50

# First to build the encoder
#  input is 784
# layer one is 256
# 输出两个向量，1个是mean,一个是var，对应一个两层的正太分布的子空间
# 其中，mean 与Var 都是 two dim
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# 随机抽取一个batch_size的latent_vector
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder=Model(x,z)

# 定义decoder
input_decoder=Input(K.int_shape(z)[1:])
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(input_decoder)
x_decoded_mean = decoder_mean(h_decoded)
decoders=Model(input_decoder,x_decoded_mean)
z_decoded = decoders(z)

# 定义Loss funtion
def vae_loss(x, x_decoded_mean):

    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

# 定义model
vae = Model(x, z_decoded)
vae.compile(optimizer='rmsprop', loss=vae_loss)

x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# training
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

vae.save("fashion_vae.md5")
decoders.save("decoders_fashion.md5")


