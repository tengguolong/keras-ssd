# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:43:20 2019

@author: Teng
"""

from __future__ import print_function
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from math import ceil
import numpy as np
import os
import argparse
#from matplotlib import pyplot as plt

from config import cfg
from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a SSD model')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--init', dest='init_epoch',
                        help='initial epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='train the dataset how many times',
                        default=1, type=int)
    parser.add_argument('--net', dest='base_net',
                        help='base net to extract feature',
                        default='vgg16', type=str)
    parser.add_argument('--no_weights', action='store_true',
                        default=False, dest='no_weights')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default='../keras-faster-rcnn/io/vgg16_weights_no_top.h5', type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='num images used in one update',
                        default=8, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
print('Called with args:\n', args)

 # allocate GPU resources
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import tensorflow as tf
c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
K.tensorflow_backend.set_session(sess)



img_height = 420 # Height of the model input images
img_width = 630 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [21.7, 21.7, 21.7] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 2 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales = [0.08, 0.16, 0.32, 0.64, 0.96]
aspect_ratios = [[0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0]]
two_boxes_for_ar1 = True
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True


K.clear_session() # Clear previous models from memory.
model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    divide_by_stddev=None)

print(model.summary())


if not args.no_weights:
    model.load_weights(args.weights, by_name=True)


# 3: Instantiate an optimizer and the SSD loss function and compile the model.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

# The directories that contain the images.
images_dir      = '../data/ghc/nuts/JPEGImages/'
# The directories that contain the annotations.
annotations_dir      = '../data/ghc/nuts/Annotations/'
# The paths to the image sets.
train_image_set_filename    = '../data/ghc/nuts/ImageSets/Main/train.txt'
val_image_set_filename      = '../data/ghc/nuts/ImageSets/Main/val.txt'
# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = cfg.CLASSES
train_dataset.parse_xml(images_dirs=[images_dir],
                        image_set_filenames=[train_image_set_filename],
                        annotations_dirs=[annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

batch_size = args.batch_size

# 4: Set the image transformations for pre-processing and data augmentation options.
# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
#val_dataset_size   = val_dataset.get_dataset_size()
print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
#print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch < 80:
        return 1e-3
    elif epoch < 120:
        return 1e-4
    else:
        return 1e-5


# Define model callbacks.
# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='ssd630x420_nuts_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=10)
csv_logger = CSVLogger(filename='ssd630x420_nuts_training_log.csv',
                       separator=',',
                       append=True)
learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
terminate_on_nan = TerminateOnNaN()
callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]


# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = args.init_epoch
final_epoch     = args.epochs
steps_per_epoch = ceil(train_dataset_size/batch_size)

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              workers=1,
#                              validation_data=val_generator,
#                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)





