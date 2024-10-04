from tensorflow.python.keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.layers import *
from keras.models import *

from utils.metrics import get_metrics_results
from utils.plot_infereces import plot_inferences
from utils.process_scene_toa import *

CUDA_DEVICE = 1

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

N_CHANNELS = 10
OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/output/active_fire'

WEIGHTS_ACTIVE_FIRE_PATH = '/home/marycamila/flaresat/results_comparison/source/active_fire/model_unet_Voting_10c_final_weights.h5'
THRESHOLD_ACTIVE_FIRE = 0.25

#Flaresat

MODEL_PATH = '/home/marycamila/flaresat/train/train_output/transfer_learning/flaresat-10c-16bs-32f-3lr.hdf5'
THRESHOLD_FLARESAT = 0.50

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_height=256, input_width=256, n_filters=64, dropout=0.1, batchnorm=True):
    input_img = Input(shape=(input_height, input_width, N_CHANNELS))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2,
                      kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4,
                      kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8,
                      kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16,
                      kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8,
                      kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4,
                      kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2,
                      kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1,
                      kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


active_fire_model = get_unet()
active_fire_model.load_weights(WEIGHTS_ACTIVE_FIRE_PATH)

truth_patches = np.array([load_patch(path, N_CHANNELS, bands=N_CHANNELS) for path in images_test['tiff_file']])
truth_masks = np.array([get_mask_patch(path) for path in masks_test['mask_file']])

# Activefire Model
method_masks = active_fire_model.predict(truth_patches)

method_masks_binary = np.where(method_masks > THRESHOLD_ACTIVE_FIRE, 1, 0)

y_test_flat = truth_masks.flatten()
y_pred_flat = method_masks_binary.flatten()

get_metrics_results(y_pred_flat, y_test_flat)

#Flaresat Model

model = tf.keras.models.load_model(MODEL_PATH)

y_pred = model.predict(truth_patches)
y_pred_thresholded = np.where(y_pred > THRESHOLD_FLARESAT, 1, 0)
flaresat_output = (y_pred_thresholded * 255).astype(np.uint8)

plot_inferences(truth_masks, method_masks_binary, truth_patches, flaresat_output, OUTPUT_PATH, list_entities_plot=[], method="af", n_images=len(truth_masks), cloud_masks=[])