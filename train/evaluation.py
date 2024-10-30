from sklearn.metrics import precision_score, recall_score, f1_score
import utils.processing as processing
from tensorflow.keras.models import load_model
import tensorflow as tf

import os
import numpy as np
import pandas as pd

CUDA_DEVICE = 1
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
THRESHOLD = 0.50

IMAGE_SIZE = 256

N_CHANNELS = 10
BANDS = []

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

test_images = np.array([processing.load_image(path, N_CHANNELS, bands=BANDS) for path in images_test['tiff_file']])
test_masks = np.array([processing.load_mask(path) for path in masks_test['mask_file']])

model_path = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
#model_path = '/home/marycamila/flaresat/train/train_output/flaresat10c0005.hdf5'
model = load_model(model_path)

y_pred = model.predict(test_images)

bins_y_pred = y_pred.flatten()
max_value = np.max(y_pred)
print(f"Max Value Predicted: {max_value}")

y_pred_binary = np.where(y_pred > THRESHOLD, 1, 0)

y_test_flat = test_masks.flatten()
y_pred_flat = y_pred_binary.flatten()

precision = precision_score(y_test_flat, y_pred_flat)
recall = recall_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat)

intersection = np.logical_and(y_test_flat, y_pred_flat)
union = np.logical_or(y_test_flat, y_pred_flat)
iou = np.sum(intersection) / np.sum(union)

dice = dice_coefficient(y_test_flat, y_pred_flat)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Dice Coefficient: {dice}")
print(f"IoU: {iou}")
