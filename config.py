# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:08:54 2019

@author: Teng
"""

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# MISC

#__C.CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
#               'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']

__C.CLASSES = ('background', 'insulator', 'normal', 'broken',
               'nuts', 'nuts-normal', 'lost', 'loose', 'nuts-unsure')

__C.class_to_id = {'background':0, 'insulator':1, 'normal':1, 'broken':1,
                   'nuts':2, 'nuts-normal':2, 'lost':2, 'loose':2, 'nuts-unsure':2}


__C.IM_HEIGHT = 300
__C.IM_WIDTH = 300

__C.PIXEL_MEANS = [26.18, 26.18, 26.18] # insulator+nuts
#__C.PIXEL_MEANS = [123, 117, 104] # VOC







# Training options

__C.TRAIN = edict()

__C.TRAIN.BATCH_SIZE = 8