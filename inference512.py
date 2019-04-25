# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:27:54 2019

@author: Teng
"""

from __future__ import print_function
import os
import cv2
import pickle
import numpy as np
import argparse
from datetime import datetime as timer
from keras import backend as K

from models.keras_ssd512 import ssd_512
from config import cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a classifier network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()
print('Called with args:\n', args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# Set the image size.
img_height = 512
img_width = 512

K.clear_session() # Clear previous models from memory.

model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=2,
                mode='inference',
                l2_regularization=0.0005,
#                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # VOC 
                scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06], # COCO
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
               two_boxes_for_ar1=True,
               steps=[8, 16, 32, 64, 128, 256, 512],
               offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               normalize_coords=True,
               subtract_mean=cfg.PIXEL_MEANS,
               swap_channels=False,
               confidence_thresh=0.5,
               iou_threshold=0.45,
               top_k=200,
               nms_max_output_size=400)


def pred(img_path):
    input_images = [] # Store resized versions of the images here.
    orig_image = cv2.imread(img_path)
    img = cv2.resize(orig_image, (img_height, img_width)).astype('float32')
    input_images.append(img)
    input_images = np.array(input_images)
    
    
    y_pred = model.predict(input_images)
    return y_pred
    


weights_path = os.path.abspath(args.weights)
model.load_weights(weights_path, by_name=True)

with open('../data/ghc/insulator+nuts/ImageSets/Main/debug.txt') as f:
    ims = [x.strip()+'.jpg' for x in f.readlines()][0:80:2]
    
t = timer.now()
dets = []
for i in range(len(ims)):
    img_path = '../data/ghc/insulator+nuts/JPEGImages/'+ims[i]  
    dets.append(pred(img_path))
delta = timer.now() - t
print(len(ims), 'images cost: ', delta)
print('average cost: ', delta/len(ims))

with open('dets.pkl', 'wb') as f:
    pickle.dump(dets, f)

