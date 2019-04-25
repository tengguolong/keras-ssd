# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:03:56 2019

@author: Teng
"""
from __future__ import print_function
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.utils import Sequence
from keras.models import load_model
from math import ceil
import os
import cv2
import argparse

from config import cfg
from models.keras_ssd25 import ssd25
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd_simplified import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


class seq_generator(Sequence):
    def __init__(self, generator, steps=1):
        self.generator = generator
        self.steps = steps
    
    def __len__(self):
        '''Denote the number of batches per epoch'''
        return int(self.steps)

    def __getitem__(self, index):
        '''generate one batch of data'''
        return next(self.generator)

    def on_epoch_end(self):
        '''updata index after each epoch'''
        pass


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a classifier network')
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
    parser.add_argument('--load_model', action='store_true',
                        default=False, dest='load_model')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default='snapshot/vgg16_weights_for_ssd.h5', type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='num images used in one update',
                        default=4, type=int)
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



img_height = 640 # Height of the model input images
img_width = 960 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = 26.18 # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = False # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 2 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales=[0.03, 0.06, 0.12, 0.24, 0.48, 0.6, 0.75, 1.0]
print('\nUsed scales: ', scales, '\n')
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = None # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True


K.clear_session() # Clear previous models from memory.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd = SGD(lr=1e-3, momentum=0.9, decay=0.0, nesterov=False)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

if args.load_model:
    model = load_model(args.weights, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                     'L2Normalization': L2Normalization,
                                                     'compute_loss': ssd_loss.compute_loss})
else:
    model = ssd25(image_size=(img_height, img_width, img_channels),
                  n_classes=n_classes,
                  mode='training',
                  l2_regularization=0.0005,
                  scales=scales,
                  aspect_ratios_per_layer=aspect_ratios,
                  two_boxes_for_ar1=two_boxes_for_ar1,
                  steps=steps,
                  offsets=offsets,
                  clip_boxes=clip_boxes,
                  variances=variances,
                  normalize_coords=normalize_coords,
                  subtract_mean=mean_color,
                  swap_channels=swap_channels)
    
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    
    if not args.no_weights:
        model.load_weights(args.weights, by_name=True)
        

print(model.summary())


# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)


# The directories that contain the images.
images_dir      = '../data/ghc/insulator+nuts/JPEGImages/'
# The directories that contain the annotations.
annotations_dir      = '../data/ghc/insulator+nuts/Annotations/'

# The paths to the image sets.
train_image_set_filename    = '../data/ghc/insulator+nuts/ImageSets/Main/train.txt'
val_image_set_filename      = '../data/ghc/insulator+nuts/ImageSets/Main/debug.txt'

# The XML parser needs to know what object class names to look for and in which order to map them to integers.
classes = cfg.CLASSES

train_dataset.parse_xml(images_dirs=[images_dir],
                        image_set_filenames=[train_image_set_filename],
                        annotations_dirs=[annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[images_dir],
                      image_set_filenames=[val_image_set_filename],
                      annotations_dirs=[annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)


batch_size = args.batch_size # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.
# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height,
                width=img_width,
                interpolation_mode = cv2.INTER_AREA)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.

predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

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
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)



train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
#                                         transformations=[ssd_data_augmentation],
                                         transformations=[convert_to_3_channels, resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels, resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()
print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch < 33:
        return 1e-3
    elif epoch < 35:
        return 2e-4
    else:
        return 1e-4


# Define model callbacks.
# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='snapshot/ssd25_UsingAug_epoch-{epoch:02d}_loss-{loss:.3f}_val_loss-{val_loss:.3f}.h5',
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1) 

csv_logger = CSVLogger(filename='log/ssd25_UsingAug_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
terminate_on_nan = TerminateOnNaN()
callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]


initial_epoch   = args.init_epoch
final_epoch     = args.epochs
steps_per_epoch = ceil(train_dataset_size/batch_size)
val_steps = ceil(val_dataset_size/batch_size)
'''
history = model.fit_generator(generator=seq_generator(train_generator, steps_per_epoch),
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              workers=1,
                              max_queue_size = 8,
                              use_multiprocessing=True,
                              validation_data=seq_generator(val_generator, val_steps),
                              validation_steps=val_steps,
                              initial_epoch=initial_epoch)
'''
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=val_steps,
                              initial_epoch=initial_epoch)

