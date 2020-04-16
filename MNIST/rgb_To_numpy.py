# -*- coding: UTF-8 -*-
'''
加载图片，转为数组
'''

from __future__ import print_function
import numpy as np
#可视化
import matplotlib.pyplot as plt
#图片转换
from PIL import Image

#方法1
img = Image.open('./dataset/number1.png')
img_1 = img.convert('1')
array_img = np.array(img_1)
array_img = array_img.reshape(1, 784)
# 规范化
array_img /= 255

np.savez('number.npz', array_img)

#方法2
#读取图片并转换为数组
img_path = './dataset/number1.png'
img = image.load_img(img_path, target_size=(28, 28))
x = image.img_to_array(img)
x = x.reshape(3, 784)
# 规范化
x /= 255