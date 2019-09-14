__author__ = "Luke Liu"
#encoding="utf-8"
import keras
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.models import  Sequential,Input,Model

import pickle
import numpy as np
import  os
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
paths_labels=unpickle("AM_S")
images_path = paths_labels['images_path']
labels=paths_labels['scores_path']
#scores rank 预处理
scores=[round(label,2) for label in labels]

images_array=np.zeros((2000,224,224,3),dtype=np.float32)
for i in range(2000):
    img=Image.open(images_path[i])
    img=img.resize((224,224))
    image=np.asarray(img)
    im=image/255.
    images_array[0,:,:,:]+=im
    print("finis {}/2000".format(i+1))
print(images_array.shape)

model=Sequential()
model.add(Conv2D(64,(3,3),padding="SAME",activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),padding="SAME",activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128,(3,3),padding="SAME",activation="relu"))
model.add(Conv2D(128,(3,3),padding="SAME",activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(3,3),padding="SAME",activation="relu"))
model.add(Conv2D(256,(3,3),padding="SAME",activation="relu"))
model.add(Conv2D(256,(3,3),padding="SAME",activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Conv2D(512,(3,3),padding="SAME",activation="relu"))
model.add(Conv2D(512,(3,3),padding="SAME",activation="relu"))
model.add(Conv2D(512,(3,3),padding="SAME",activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Conv2D(512,(3,3),padding="SAME",activation="relu"))
model.add(Conv2D(512,(3,3),padding="SAME",activation="relu"))
model.add(Conv2D(512,(3,3),padding="SAME",activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(2048,activation='relu'))
model.add(Dense(2048,activation="relu"))
model.add(Dense(1,activation='relu'))
model.summary()
from keras import optimizers
from keras import losses
model.compile(loss='mse',
 optimizer=optimizers.RMSprop(lr=1e-4),
 metrics=['acc'])

history=model.fit(images_array,scores,
                  batch_size=2,
                  epochs=1,
                  )

model.save("123.md5")