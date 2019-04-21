# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:27:54 2019

@author: Teng
"""

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import os
import cv2
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from datetime import datetime as timer

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from config import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Set the image size.
img_height = 300
img_width = 300



K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=2,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=cfg.PIXEL_MEANS,
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)


def pred(img_path):
    input_images = [] # Store resized versions of the images here.
    orig_image = cv2.imread(img_path)
    orig_image = orig_image[:, :, (2,1,0)]
    img = cv2.resize(orig_image, (img_height, img_width)).astype('float32')
    input_images.append(img)
    input_images = np.array(input_images)
    
    
    y_pred = model.predict(input_images)
    return y_pred
    


weights_path = 'snapshot/ssd300_no_aug_epoch-10_loss-6.752_val_loss-7.262.h5'
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

