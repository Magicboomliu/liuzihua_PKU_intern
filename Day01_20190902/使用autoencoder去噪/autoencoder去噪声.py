__author__ = "Luke Liu"
#encoding="utf-8"
import keras
import tensorboard as tb
import tensorflow as tf
from keras.datasets import  mnist
import numpy  as np
import matplotlib.pyplot as plt
from PIL import  Image
(x_train,y_train),(x_test,y_test)= mnist.load_data()
# change the data shape
x_train =x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

# change it into network shape
x_train=np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


# add some  white nosie
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# plt.imshow(x_train_noisy[1].reshape(28,28,),cmap='gray')
# plt.show()

# 定义keras模型
from keras.layers import  Dense,Conv2D,UpSampling2D,MaxPool2D
from keras.optimizers import rmsprop
from  keras.models import  Model,Input

# 首先定义encoder
input_image=Input((28,28,1),name="Encoder_input",dtype=np.float32)

x=Conv2D(32,(3,3),padding="SAME",activation='relu')(input_image)
x=MaxPool2D((2,2),padding="SAME")(x)
x=Conv2D(32,(3,3),padding="SAME",activation='relu')(x)
encoder_output = MaxPool2D((2,2),padding="SAME")(x)

encoder = Model(input_image,encoder_output)
encoder.summary()

# 其次定义 decoder
x=Conv2D(32,(3,3),padding="SAME",activation='relu')(encoder_output)
x=UpSampling2D((2,2))(x)
x=Conv2D(32,(3,3),padding="SAME",activation='relu')(x)
x=UpSampling2D((2,2))(x)
decoder_output = Conv2D(1,(3,3),padding="same",activation='sigmoid')(x)

#损失函数定义为交叉熵
autoencoder = Model(input_image,decoder_output)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

autoencoder.save('autoencoder.h5')


