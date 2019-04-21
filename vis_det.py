# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:06:16 2019

@author: Teng
"""

import os
import re
import cv2
import numpy as np
import matplotlib
import pickle
matplotlib.use('Agg')
from matplotlib import pyplot as plt

img_height = 300
img_width = 300
confidence_threshold = 0.5
classes = ('background', 'insulator', 'nuts')

def get_im_name(im_path, imdb):
    if imdb == 'voc2007':
        p = r'\d{6}\.jpg'
    elif imdb == 'kitty':
        p = r'\d{6}\.png'
    elif imdb in ['insulator', 'nuts', 'detect']:
        p = '[0-9A-Z\_]{1,100}\.jpg'
    find = re.search(p, im_path)
    return find.group(0)


def vis_dets(img_path, y_pred):
    orig_image = cv2.imread(img_path)
    orig_image = orig_image[:, :, (2,1,0)]
    
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    
    y_pred_thresh[0][:, 2:] = np.clip(y_pred_thresh[0][:, 2:], 0, img_height)
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])
    
    
    # Display the image and draw the predicted boxes onto it.
    
    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()
    
    fig = plt.figure()
    fig.set_size_inches(orig_image.shape[1]/400.0, orig_image.shape[0]/400.0)
    plt.imshow(orig_image)
    
    current_axis = plt.gca()
    
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_image.shape[1] / img_width
        ymin = box[3] * orig_image.shape[0] / img_height
        xmax = box[4] * orig_image.shape[1] / img_width
        ymax = box[5] * orig_image.shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
    name = get_im_name(img_path, 'detect')
    print(name)
    plt.savefig('results/'+str(name), quality=95)
    plt.close()
    

with open('../data/ghc/insulator+nuts/ImageSets/Main/debug.txt') as f:
    ims = [x.strip()+'.jpg' for x in f.readlines()]

with open('dets.pkl', 'rb') as f:
    dets = pickle.load(f, encoding='bytes')


for i in range(len(ims)):
    y_pred = dets[i]
    img_path = '../data/ghc/insulator+nuts/JPEGImages/'+ims[i]  
    vis_dets(img_path, y_pred)
