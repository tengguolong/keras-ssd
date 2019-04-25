# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:36:41 2019

@author: Teng
"""

from __future__ import print_function
import cv2
import numpy as np
from PIL import Image
from datetime import datetime as timer

tar = (900, 600)


t = timer.now()
ori = cv2.imread('50.jpg')
print('opencv read: ', timer.now() - t, '\n')

t = timer.now()
with Image.open('50.jpg') as f:
    img = np.array(f, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
print('PIL read: ', timer.now() - t, '\n')

t = timer.now()
img = cv2.resize(ori, tar, interpolation=cv2.INTER_NEAREST)
print('NEAREST: ', timer.now() - t, '\n')

t = timer.now()
img = cv2.resize(ori, tar, interpolation=cv2.INTER_LINEAR)
print('LINEAR: ', timer.now() - t, '\n')

t = timer.now()
img = cv2.resize(ori, tar, interpolation=cv2.INTER_CUBIC)
print('CUBIC: ', timer.now() - t, '\n')

t = timer.now()
img = cv2.resize(ori, tar, interpolation=cv2.INTER_AREA)
print('AREA: ', timer.now() - t, '\n')

t = timer.now()
img = cv2.resize(ori, tar, interpolation=cv2.INTER_LANCZOS4)
print('LANCZOS4: ', timer.now() - t, '\n')

t = timer.now()
image = cv2.cvtColor(ori, cv2.COLOR_RGB2HSV)
print('RGB2HSV: ', timer.now() - t, '\n')

t = timer.now()
image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
print('RGB2HSV: ', timer.now() - t, '\n')





