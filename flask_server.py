# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:05:17 2019

@author: Teng
"""
from __future__ import print_function, division
import os
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
from flask import Flask
from flask import jsonify, request, make_response
import json
import base64
import argparse
from datetime import datetime as timer

from test import create_models, defect_inspection

img_width = 960
img_height = 640
conf_thresh = 0.5

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='4C Detector Service')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--port', dest='service_port',
                        default=7070, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
print('Called with args:\n', args)

 # allocate GPU resources
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
K.tensorflow_backend.set_session(sess)

# SSD定位
def locate(model, image):
    img = cv2.resize(image, (img_width, img_height))
    img = np.expand_dims(img, 0)
    img = img.astype('float32')
    loc = model.predict(img)[0]
    scores = loc[:, 1]
    inds = np.where(scores > conf_thresh)[0]
    loc = np.expand_dims(loc[inds], 0)
    # 将预测的框映射到原图上
    loc[:, :, (2, 4)] *= float(image.shape[1]) / img_width
    loc[:, :, (3, 5)] *= float(image.shape[0]) / img_height
    return loc


app = Flask(__name__)
app.service_port = args.service_port
app.models = create_models()


@app.route('/')
def index():
    print('')
    return 'connected'

@app.route('/detect', methods=['POST'])
def run_detect():
#    coming_data = request.data
    # print(input_params)
    # print(json.dumps(input_params))
    # print(input_params['id'])
    input_file = request.files.get('image')
    if input_file is None:
        print("no input file.")
        return jsonify({'err': 'no input file.'})

    print("received an image")
    impy = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
    im = cv2.imdecode(impy, cv2.IMREAD_COLOR)
    print('decoded')
    print(im.shape)
    # 定位
    t = timer.now()
    loc = locate(app.models[0], im)
    print('Locating cost: ', timer.now() - t)
    # 检测
    t = timer.now()
    res = defect_inspection(app.models[1], app.models[2], im, loc)
    print('Inspecting cost: ', timer.now() - t)
    return jsonify({'loc': list(loc.ravel().astype('str')), 'res': list(res.ravel().astype('str'))})


if __name__ == '__main__':
  # 这里必须threaded=False，否则多线程中的cuda Context会有问题
    app.run(debug=False,
            use_reloader=False,
            host='0.0.0.0',
            port=app.service_port,
            threaded=False)
