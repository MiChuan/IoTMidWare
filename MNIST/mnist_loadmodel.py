# -*- coding: UTF-8 -*-
'''
加载已训练好的一个简单的深度神经网络。
加载图片，识别手写数字，输出识别结果
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility 为了重现结果

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.preprocessing import image
#加载本地数据
import loadfile as ld
from keras.models import load_model
from keras.utils import plot_model
#可视化
import matplotlib.pyplot as plt
#图片转换
from PIL import Image

#加载本地模型
model = load_model('my_model.h5')

#读取图片并转换为数组
img = Image.open('./dataset/number6.png')
img_1 = img.convert('L')
array_img = np.array(img_1)
array_img = array_img.reshape(1, 784)
print(array_img)
print('\n')
# 规范化
#array_img /= 255

#预测图片数字
pred = model.predict(array_img)
#输出预测结果
print('predicted:', pred)