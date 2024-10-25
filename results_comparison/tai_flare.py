from tensorflow.python.keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from utils.metrics import get_metrics_results
from utils.plot_infereces import plot_inferences
from utils.process_scene_toa import *
from methods.comparison_methods import get_toa_tai

CUDA_DEVICE = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

OUTPUT_DIR = '/home/marycamila/flaresat/train/train_output'
MODEL_FILE_NAME = 'flaresat.hdf5'
N_CHANNELS = 10
THRESHOLD = 0.50

OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/output/tai'

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/comparison/images_test_fire.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/comparison/masks_test_fire.csv')

cloud_masks = []

# TAI gas flare detection reference - https://www.sciencedirect.com/science/article/pii/S1569843222002631
def get_toa_tai_patch(file_path):
    img = get_toa_patch(file_path, 'REFLECTANCE')    
    clear_cloud_mask = get_cloud_mask(file_path)
    cloud_masks.append(clear_cloud_mask)
    
    flagged_pixels = get_toa_tai(clear_cloud_mask, img)

    return flagged_pixels


method_masks = np.array([get_toa_tai_patch(path) for path in images_test['tiff_file']])
truth_masks = np.array([get_mask_patch(path) for path in masks_test['mask_file']])
 
y_pred_flat = method_masks.flatten()
y_test_flat = truth_masks.flatten()

get_metrics_results(y_pred_flat,y_test_flat)

#Getting flaresat output
# truth_patches = np.array([load_patch(path, n_channels=N_CHANNELS) for path in images_test['tiff_file']])

# model_path = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
# model = tf.keras.models.load_model(model_path)

# y_pred = model.predict(truth_patches)
# y_pred_thresholded = np.where(y_pred > THRESHOLD, 1, 0)
# flaresat_output = (y_pred_thresholded * 255).astype(np.uint8)

# plot_inferences(truth_masks, method_masks, truth_patches, flaresat_output, OUTPUT_PATH, list_entities_plot=[], method="tai", n_images=len(truth_masks), cloud_masks=cloud_masks)