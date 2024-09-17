from sklearn.metrics import precision_score, recall_score, f1_score
from processing import get_img_arr, get_mask_arr
from tensorflow.keras.models import load_model
import tensorflow as tf

import os
import numpy as np
import pandas as pd

OUTPUT_DIR = '/home/marycamila/flaresat/train/train_output'
MODEL_FILE_NAME = 'flaresat.hdf5'
THRESHOLD = 0.50

IMAGE_SIZE = 256
N_CHANNELS = 10

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

test_images = np.array([get_img_arr(path, N_CHANNELS) for path in images_test['tiff_file']])
test_masks = np.array([get_mask_arr(path) for path in masks_test['mask_file']])

model_path = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
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

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"IoU: {iou}")
