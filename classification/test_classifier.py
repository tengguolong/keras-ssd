# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:01:42 2019

@author: Teng
"""

from __future__ import print_function
import sys
if __name__ == '__main__' and not '..' in sys.path:
    sys.path.append('..')
from datetime import datetime as timer
import os
import os.path as osp
import pickle
import cv2
import numpy as np
from keras.models import load_model


conf_thresh = 0.5
ssd_image_height = 640
ssd_image_width = 960
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def defect_inspection(model_insulator, model_nuts, image, det):
    # shape of det: (n, 6)  2~5: boxes  0:class, 1 or 2  1:score
    orig_scores = det[:, 1]
    inds = np.where(orig_scores > conf_thresh)[0]
    det = det[inds]
    classes = det[:, 0]
    BB = det[:, 2:]
    H, W = image.shape[:2]
    # 需要将SSD的预测框映射到原图上
#    BB[:, (0, 2)] *= W / ssd_image_width
#    BB[:, (1, 3)] *= H / ssd_image_height
    
    # insulator
    BB1 = BB[np.where(classes==1)[0]] # 原始定位结果
    B1 = np.zeros_like(BB1, dtype='float32') # 分类用的框（有0.05的裕量）
    if BB1.size > 0:
        w = BB1[:, 2] - BB1[:, 0]
        h = BB1[:, 3] - BB1[:, 1]
        B1[:, 0] = BB1[:, 0] - 0.05 * w
        B1[:, 1] = BB1[:, 1] - 0.05 * h
        B1[:, 2] = BB1[:, 2] + 0.05 * w
        B1[:, 3] = BB1[:, 3] + 0.05 * h
        B1[:, (0, 2)] = np.clip(B1[:, (0, 2)], 0, W)
        B1[:, (1, 3)] = np.clip(B1[:, (1, 3)], 0, H)
        B1 = B1.astype('int32')
#        from IPython import embed; embed()
        crops = []
        for bb in B1:
            crop = image[bb[1]:bb[3]+1, bb[0]:bb[2]+1]
            crop = crop.astype('float32')
            crop -= 52.23
            crop = cv2.resize(crop, (224, 224))
            crops.append(crop)
        crops = np.array(crops, dtype='float32')
        out = model_insulator.predict(crops) # (n, 2)
        res = np.argmax(out, axis=1)[:, np.newaxis] # (n, 1)
        scores = np.max(out, axis=1)[:, np.newaxis] # (n, 1)
        B1 = np.hstack([res, scores, BB1])
        
    else:
        B1 = np.zeros(shape=(0, 6), dtype='float32')
        
    # nuts
    BB2 = BB[np.where(classes==2)[0]]
    B2 = np.zeros_like(BB2, dtype='float32')
    if BB2.size > 0:
        w = BB2[:, 2] - BB2[:, 0]
        h = BB2[:, 3] - BB2[:, 1]
        B2[:, 0] = BB2[:, 0] - 0.05 * w
        B2[:, 1] = BB2[:, 1] - 0.05 * h
        B2[:, 2] = BB2[:, 2] + 0.05 * w
        B2[:, 3] = BB2[:, 3] + 0.05 * h
        B2[:, (0, 2)] = np.clip(B2[:, (0, 2)], 0, W)
        B2[:, (1, 3)] = np.clip(B2[:, (1, 3)], 0, H)
        B2 = B2.astype('int32')
        crops = []
        for bb in B2:
            crop = image[bb[1]:bb[3]+1, bb[0]:bb[2]+1]
            crop = crop.astype('float32')
            crop -= 86.68
            crop = cv2.resize(crop, (224, 224))
            crops.append(crop)
        crops = np.array(crops, dtype='float32')
        out = model_nuts.predict(crops) # (n, 2)
        res = np.argmax(out, axis=1)[:, np.newaxis] + 2 # (n, 1)
        scores = np.max(out, axis=1)[:, np.newaxis] # (n, 1)
#        for i in range(len(res)):
#            if res[i][0] > 2.0 and scores[i][0] < 0.8:
#                res[i] = 2.0        
        B2 = np.hstack([res, scores, BB2])
    else:
        B2 = np.zeros(shape=(0, 6), dtype='float32')
        
    B = np.vstack([B1, B2])
    # 增加一个维度，与SSD的结果保持一致，方便vis_dets函数
    B = np.expand_dims(B, 0)
    return B
    

print('loading models...')
t = timer.now()
model1 = load_model('snapshot/insulator-trainval_resnet50_epoch-27_acc-0.982_val_acc-0.993.h5',
                    compile=False)
model2 = load_model('snapshot/nuts-trainval_resnet50_epoch-20_acc-0.977_val_acc-0.977.h5',
                    compile=False)
delta = timer.now() - t
print('models loaded, cost', delta)

with open('../det.pkl', 'rb') as f:
    dets = pickle.load(f)
data_dir = '../../data/ghc/insulator+nuts'
image_dir = osp.join(data_dir, 'JPEGImages')
imagesetfile = osp.join(data_dir, 'ImageSets/Main/val.txt')
with open(imagesetfile) as f:
    image_ids = [x.strip() for x in f.readlines()][:20]

start = timer.now()
inspect = []
for i in range(len(image_ids)):
    print(image_ids[i])
    det = dets[i][0]
    im = cv2.imread(osp.join(image_dir, image_ids[i]+'.jpg'))
    t = timer.now()
    res = defect_inspection(model1, model2, im, det)
    print('defect inspection cost', timer.now()-t)
    inspect.append(res)
end = timer.now()
delta = end - start
print(len(image_ids), 'cost:', delta)
print('average cost: ', delta / len(image_ids))

with open('../cls.pkl', 'wb') as f:
    pickle.dump(inspect, f)

    
    
    