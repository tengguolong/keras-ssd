# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:23:08 2019

@author: Teng
"""

from __future__ import print_function
import os
import pickle
import numpy as np
import argparse
from datetime import datetime as timer
from keras import backend as K

from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator
from models.keras_ssd25 import ssd25
from config import cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a SSD model')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--weights', dest='weights',
                        help='model weights dir',
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()
print('Called with args:\n', args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


img_height = 640
img_width = 960
scales=[0.03, 0.06, 0.12, 0.24, 0.48, 0.6, 0.75, 1.0]
n_classes = 2

K.clear_session() # Clear previous models from memory.
model = ssd25(image_size=(img_height, img_width, 3),
              n_classes=n_classes,
              mode='inference',
              l2_regularization=0.0005, 
              scales=scales,
              aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                       [1.0, 2.0, 0.5],
                                       [1.0, 2.0, 0.5]], 
               two_boxes_for_ar1=True,
               steps=None,
               offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               normalize_coords=True,
               subtract_mean=cfg.PIXEL_MEANS,
               swap_channels=False,
               confidence_thresh=0.5,
               iou_threshold=0.3,
               top_k=200,
               nms_max_output_size=400)

weights_path = os.path.abspath(args.weights)
model.load_weights(weights_path, by_name=True)


dataset = DataGenerator()
# The directories that contain the images.
images_dir      = '../data/ghc/insulator+nuts/JPEGImages/'
# The directories that contain the annotations.
annotations_dir      = '../data/ghc/insulator+nuts/Annotations/'

# The paths to the image sets.
image_set_filename    = '../data/ghc/insulator+nuts/ImageSets/Main/val.txt'

# The XML parser needs to know what object class names to look for and in which order to map them to integers.
classes = cfg.CLASSES

dataset.parse_xml(images_dirs=[images_dir],
                  image_set_filenames=[image_set_filename],
                  annotations_dirs=[annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode='inference')

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

with open('ssd_eval_results.pkl', 'wb') as f:
    pickle.dump(results, f)

mean_average_precision, average_precisions, precisions, recalls = results
for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))



