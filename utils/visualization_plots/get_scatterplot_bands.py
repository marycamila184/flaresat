import glob
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re

import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.layers import *
from keras.models import *
import tifffile as tiff

MAX_PIXEL_VALUE = 65535
CUDA_DEVICE = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

N_CHANNELS = 10
THERMAL_BAND = '10'

PATH_MTL_SCENE = '/media/marycamila/Expansion/raw/2019'
PATH_MTL_VOLCANOES = '/media/marycamila/Expansion/raw/volcanoes'
PATH_MTL_FIRE = '/media/marycamila/Expansion/raw/active_fire/metadata'

# Activefire
WEIGHTS_ACTIVE_FIRE_PATH = '/home/marycamila/flaresat/results_comparison/source/active_fire/model_unet_Voting_10c_final_weights.h5'
THRESHOLD_ACTIVE_FIRE = 0.25

#Flaresat

MODEL_PATH = '/home/marycamila/flaresat/train/train_output/transfer_learning/flaresat-10c-16bs-32f-3lr.hdf5'
THRESHOLD_FLARESAT = 0.50
