__author__ = "Luke Liu"
#encoding="utf-8"

'''Neural style transfer with Keras.
Run the script with:
```
python neural_style_transfer.py path_to_your_base_image.jpg \
    path_to_your_reference.jpg prefix_for_results
```
e.g.:
```
python neural_style_transfer.py img/tuebingen.jpg \
    img/starry_night.jpg results/my_result
```
Optional parameters:
```
--iter, To specify the number of iterations \
    the style transfer takes place (Default is 10)
--content_weight, The weight given to the content loss (Default is 0.025)
--style_weight, The weight given to the style loss (Default is 1.0)
--tv_weight, The weight given to the total variation loss (Default is 1.0)
```
It is preferable to run this script on GPU, for speed.
Example result: https://twitter.com/fchollet/status/686631033085677568
# Details
Style transfer consists in generating an image
with the same "content" as a base image, but with the
"style" of a different picture (typically artistic).
This is achieved through the optimization of a loss function
that has 3 components: "style loss", "content loss",
and "total variation loss":
- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence.
- The style loss is where the deep learning keeps in --that one is defined
using a deep convolutional neural network. Precisely, it consists in a sum of
L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from
different layers of a convnet (trained on ImageNet). The general idea
is to capture color/texture information at different spatial
scales (fairly large scales --defined by the depth of the layer considered).
 - The content loss is a L2 distance between the features of the base
image (extracted from a deep layer) and the features of the combination image,
keeping the generated image close enough to the original one.
# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''

from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
from keras.applications import vgg19
from keras import backend as K

# parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
#
# parser.add_argument('base_image_path', metavar='base', type=str,
#                     help='Path to the image to transform.')
# parser.add_argument('style_reference_image_path', metavar='ref', type=str,
#                     help='Path to the style reference image.')
# parser.add_argument('result_prefix', metavar='res_prefix', type=str,
#                     help='Prefix for the saved results.')
# parser.add_argument('--iter', type=int, default=10, required=False,
#                     help='Number of iterations to run.')
# parser.add_argument('--content_weight', type=float, default=0.025, required=False,
#                     help='Content weight.')
# parser.add_argument('--style_weight', type=float, default=1.0, required=False,
#                     help='Style weight.')
# parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
#                     help='Total Variation weight.')
#
# args = parser.parse_args()
# # base_image_path = args.base_image_path
# style_reference_image_path = args.style_reference_image_path
# result_prefix = args.result_prefix
# iterations = args.iter

# 定义image的位置
base_image_path = 'Test_pictures/content.jpg'
style_reference_image_path = 'Test_pictures/style_image.jpg'
result_prefix = 'result_prefix'
iterations = 200

# these are the weights of the different loss components
# total_variation_weight = args.tv_weight
# style_weight = args.style_weight
# content_weight = args.content_weight

total_variation_weight=1.0
style_weight=1.0
content_weight=0.025
# dimensions of the generated picture.
width, height = load_img(base_image_path).size
# 读取原始图片的大小,并且按照比例尺寸进行调整
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# util function to open, resize and format pictures into appropriate tensors

# 处理图片，给图片加上一维度（个数），并且将输入调整为vgg16能够接受的尺寸，返回imgae,array
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image
# 主要是去掉num维度
def deprocess_image(x):
# 如果使用时theano ,则组织成theano读取数据的形式(num,height,width,channel)

    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
# 如果是TensorFlow,就组织成默认的（height,width,channels）的形式
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
#去掉zero_cenetr
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]

# 进行元素替换，限制在0,255之间
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# get tensor representations of our images
# 转化为TF向量，base_image
base_image = K.variable(preprocess_image(base_image_path))
# 转化TF向量，style_image
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# this will contain our generated image
# 定义输入的图片，placeholder
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
#将三张图片混合输入，作为input_sensor
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)
# 此时的shape is（3,img_nrows,img_ncols)

# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights

# 不包含全连接层的vgg16
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
#将每一层的layer_name与每一层的output形成一个字典

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions
# the gram matrix of an image tensor (feature-wise outer product)

# 构建一个gram矩阵，用来表示特征图的互相关
def gram_matrix(x):
    assert K.ndim(x) == 3 #s深度为3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

#风格损失，输入style和combination
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    # 分别计算gram矩阵
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

# 内容损失，直接用rmse
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent

# 定义总迁移损失
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:

        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))

# combine these loss functions into a single scalar
loss = K.variable(0.0)
#第5个block的第2个卷积层
layer_features = outputs_dict['block5_conv2']
# 0 is base,1 is style, and 2 is combination
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
# 计算内容损失
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

for layer_name in feature_layers:
    #提取一层的特征
    layer_features = outputs_dict[layer_name]
    # 1为style图的特征
    style_reference_features = layer_features[1, :, :, :]
    # 2为 combination的特征
    combination_features = layer_features[2, :, :, :]
    # 计算风格损失
    sl = style_loss(style_reference_features, combination_features)
    # 将风格损失加入总损失中
    loss += (style_weight / len(feature_layers)) * sl
# 最后在加入一个total 迁移损失
loss += total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
# 计算梯度
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)
# 定义一个function,输入combination_image,输出（loss+grad)
f_outputs = K.function([combination_image], outputs)

# 计算损失与input
def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:

        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
# outs的一个输入为loss
    loss_value = outs[0]

    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
# 预处理一张image

x = preprocess_image(base_image_path)
# 开始循环
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)

    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))