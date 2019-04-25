# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:06:16 2019

@author: Teng
"""

from __future__ import print_function
import re
import cv2
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

img_height = 640
img_width = 960
confidence_threshold = 0.5
#classes = ('background', 'insulator', 'nuts')
classes = ('normal', 'broken', 'normal', 'lost', 'loose', 'obscured')
#colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()
# R G B order
colors = [[0.0, 0.5, 1.0, 1],
          [0.9, 0.1, 0, 1],
          [0.0, 1.0, 0.6, 1],
          [0.82, 1, 0, 1],
          [0, 0.7, 1, 1],
          [0.7, 0.7, 0.7, 1]]

def get_im_name(im_path, imdb):
    if imdb == 'voc2007':
        p = r'\d{6}\.jpg'
    elif imdb in ['insulator', 'nuts', 'detect']:
        p = '[0-9a-zA-Z\_\-]{1,100}\.jpg'
    find = re.search(p, im_path)
    return find.group(0)


def vis_dets(img_path, y_pred):
    name = get_im_name(img_path, 'insulator')
    name = name[:-4] + '.jpg'
    print('\n\n', name)
    orig_image = cv2.imread(img_path)
    orig_image = orig_image[:, :, (2,1,0)]
    
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    
    
#    np.set_printoptions(precision=2, suppress=True, linewidth=90)
#    print("Predicted boxes:\n")
#    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])
    
    fig = plt.figure()
    fig.set_size_inches(orig_image.shape[1] / 100, orig_image.shape[0] / 100)
    linewidth = 2 * orig_image.shape[1] / 1200
    font_size = linewidth * 7
    plt.imshow(orig_image)
    
    current_axis = plt.gca()
    
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=linewidth))  
        current_axis.text(xmin, ymin, label, size=font_size, color='white', bbox={'facecolor':color, 'alpha':1})
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
    plt.savefig('results/inspection-val/'+str(name), quality=95)
    plt.close()
    

with open('../data/ghc/insulator+nuts/ImageSets/Main/val.txt') as f:
    ims = [x.strip()+'.jpg' for x in f.readlines()][:20]

with open('cls.pkl', 'rb') as f:
    dets = pickle.load(f, encoding='bytes')


for i in range(20):
    y_pred = dets[i]
    img_path = '../data/ghc/insulator+nuts/JPEGImages/'+ims[i]  
    vis_dets(img_path, y_pred)
    

