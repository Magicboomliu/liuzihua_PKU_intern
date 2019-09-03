__author__ = "Luke Liu"
#encoding="utf-8"
import keras
from keras.datasets import  mnist
from keras.models import load_model
import numpy as np
(x_train, _),(x_test,y_test)=mnist.load_data()

# change the data shape
x_train =x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

# change it into network shape
x_train=np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# add the noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

import matplotlib.pyplot as plt

model=load_model("autoencoder.h5")
pre=model.predict(x_test_noisy)
fig = plt.gcf()
plt.subplot(121)
plt.imshow(x_test_noisy[2].reshape(x_test_noisy.shape[1],x_test_noisy.shape[2],),cmap='gray')
plt.subplot(122)
a=plt.imshow(pre[2].reshape(pre.shape[1],pre.shape[2],),cmap='gray')
plt.show()
fig.savefig("test.jpg")