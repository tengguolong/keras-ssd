# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:24:48 2019

@author: Teng
"""

import cv2
import numpy as np

mean_color = 26

def resize(img, factor):
    H = img.shape[0]
    W = img.shape[1]
    if np.abs(factor - 1) < 0.001:
        return img
    rs = cv2.resize(img, None, fx=factor, fy=factor)
    H1 = rs.shape[0]
    W1 = rs.shape[1]
    ret = mean_color * np.ones_like(img, dtype=np.uint8)
    ymin = H1 // 2 - H // 2 
    xmin = W1 // 2 - W // 2 
    if factor > 1:
        ret = rs[ymin:ymin+H, xmin:xmin+W, :]
    else:
        ret[-ymin:-ymin+H1, -xmin:-xmin+W1, :] = rs[:, :, :]
    return ret

def translate(img, offset):
    H = img.shape[0]
    W = img.shape[1]
    x = int(np.round(np.abs(W * offset)))
    y = int(np.round(np.abs(H * offset)))
    canvas = mean_color * np.ones(shape=(H+y, W+x, 3), dtype=np.uint8)
    ret = np.zeros_like(img, dtype=np.uint8)
    if offset > 0.001:
        canvas[y:, x:, :] = img[:, :, :]
        ret[:, :, :] = canvas[:H, :W, :]
    elif offset < -0.001:
        canvas[:H, :W, :] = img[:, :, :]
        ret[:, :, :] = canvas[y:, x:, :]
    else:
        ret = img
    return ret
    
def rotate(img, angle):
    if np.abs(angle) < 1:
        return img
    H = img.shape[0]
    W = img.shape[1]
    ctr = (H//2, W//2)
    scale = 0.707 / np.sin(0.25*np.pi+np.abs(angle)*np.pi/180)
    M = cv2.getRotationMatrix2D(ctr, angle, scale)
    rotated = cv2.warpAffine(img, M, (W, H))
    return rotated
        

def contrast_trans(img, contrast):
    if np.abs(contrast - 1.0) < 0.01:
        return img
    ret = np.array(img, 'float32')
    ret = np.clip(127.5 + contrast * (img - 127.5), 0, 255).astype('uint8')
    return ret

def gamma_trans(img, gamma):
    if np.abs(1.0 - gamma) < 0.01:
        return img
    gamma_inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def brightness_trans(img, brightness):
    if np.abs(brightness) < 0.1:
        return img
    ret = np.array(img, dtype='float32')
    ret += brightness
    ret = np.clip(ret, 0, 255).astype('uint8')
    return ret


def transform(img, factor=1, offset=0, angle=0,
              brightness=0, contrast=1, gamma=1):
    ret = resize(img, factor)
    ret = translate(ret, offset)
    ret = rotate(ret, angle)
    ret = brightness_trans(ret, brightness)
    ret = contrast_trans(ret, contrast)
    ret = gamma_trans(ret, gamma)
    return ret
    
    
    
    