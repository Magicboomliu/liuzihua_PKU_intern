__author__ = "Luke Liu"
#encoding="utf-8"
import keras
import numpy as np
from keras.models import load_model
import  matplotlib.pyplot as plt
gen_dir='decoders_fashion.md5'
gen=load_model(gen_dir)
n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[yi, xi]])
        x_decoded = gen.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[(n - i - 1) * digit_size: (n - i) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))

plt.imshow(figure,cmap='gray')
plt.savefig('test.png')
plt.show()


