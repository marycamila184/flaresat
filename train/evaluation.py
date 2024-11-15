from sklearn.metrics import precision_score, recall_score, f1_score
import utils.processing as processing

from models.transfer_learning.unet_attention_sentinel_landcover import unet_attention_sentinel_landcover
from models.transfer_learning.unet_sentinel_landcover import unet_sentinel_landcover
from models.attention_unet import unet_attention_model
from models.unet import unet_model

import os
import numpy as np
import pandas as pd

CUDA_DEVICE = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

#MODEL_PATH = '/home/marycamila/flaresat/train/train_output/unet/flaresat-10c-32f-16bs.hdf5'
MODEL_PATH = '/home/marycamila/flaresat/train/train_output/flaresat.weights.h5'
THRESHOLD = 0.50

N_CHANNELS = 3
BANDS = [4,5,6]
DICT_CHANNELS = (4,5,6)

IMAGE_SIZE=(256,256)

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

test_images = np.array([processing.load_image(path, N_CHANNELS, bands=BANDS) for path in images_test['tiff_file']])
test_masks = np.array([processing.load_mask(path) for path in masks_test['mask_file']])


#model = unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS))
#model = unet_attention_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS))
#model = unet_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), dict_channels=DICT_CHANNELS)
model = unet_attention_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), dict_channels=DICT_CHANNELS)
model.load_weights(MODEL_PATH)

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
