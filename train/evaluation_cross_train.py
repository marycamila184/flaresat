from sklearn.metrics import precision_score, recall_score, f1_score
from train.utils.cross_split import create_folds
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

THRESHOLD = 0.50

NUM_FOLDS = 4

IMAGE_SIZE=(256,256)

SPECIFICITY = False
RANDOM_STATE = 42

model_name_map = {
    'unet': ('Unet', 0.001),
    'unet_attention': ('Attention Unet', 0.0005),
    'unet_sentinel_landcover': ('Land Cover Pre Trained Unet', 0.001),
    'unet_attention_sentinel_landcover': ('Land Cover Pre Trained Unet Attention', 0.001)
}

channels_readable = {
    '156': '2, 6 e 7',
    '456': '5, 6 e 7',
    '3456': '4, 5, 6 e 7',
    '10': '1 a 10'
}

flare_patches = pd.read_csv('dataset/flare_patches.csv')
urban_patches = pd.read_csv('dataset/urban_patches.csv')
wildfire_patches = pd.read_csv('dataset/fire_patches.csv')

images_flare, images_urban, images_wildfire = create_folds(flare_patches, urban_patches, wildfire_patches)

list_models = ["unet", "unet_attention", "unet_sentinel_landcover", "unet_attention_sentinel_landcover"]
list_bands = [[1,5,6], [4,5,6], [3,4,5,6], []] 
dict_channels = [(1,5,6), (4,5,6), (3,4,5,6), ()]
path_channels = ['156', '456', '3456', '10']

results = []

for model_name in list_models:
    pretty_model_name, learning_rate = model_name_map[model_name]

    for index, bands in enumerate(list_bands):
        channels = len(bands) if len(bands) > 0 else 10

        row = {
            'Model': pretty_model_name,
            'Learning Rate': learning_rate,
            'Channels': channels,
            'Bands': channels_readable[path_channels[index]]
        }

        for fold in range(NUM_FOLDS):
            print(f"---- Model: {model_name}, Channels: {bands} - Fold: {fold + 1}")
            
            path = f'train/train_output/cross_validation_split/{model_name}/fold_{fold+1}/fold_{fold+1}_{model_name}_b{path_channels[index]}.keras'

            if model_name == 'unet':
                model_instance = unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels))
            elif model_name == 'unet_attention':
                model_instance = unet_attention_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels))
            elif model_name == 'unet_sentinel_landcover':
                model_instance = unet_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels), dict_channels=dict_channels[index])
            else:
                model_instance = unet_attention_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels), dict_channels=dict_channels[index])

            model_instance.load_weights(path)

            flare_patches = images_flare[images_flare['fold'] == fold]['tiff_file']
            flare_masks = images_flare[images_flare['fold'] == fold]['mask_file']

            urban_patches = images_urban[images_urban['fold'] == fold]['tiff_file']
            urban_masks = images_urban[images_urban['fold'] == fold]['mask_file']

            fire_patches = images_wildfire[images_wildfire['fold'] == fold]['tiff_file']
            fire_masks = images_wildfire[images_wildfire['fold'] == fold]['mask_file']

            all_patches = pd.concat([flare_patches, urban_patches, fire_patches]).tolist()
            all_masks = pd.concat([flare_masks, urban_masks, fire_masks]).tolist()

            test_images = np.array([processing.load_image(p, channels, bands=bands) for p in all_patches])
            test_masks = np.array([processing.load_mask(m) for m in all_masks])

            y_pred = model_instance.predict(test_images)
            y_pred_binary = np.where(y_pred > THRESHOLD, 1, 0)

            y_test_flat = test_masks.flatten()
            y_pred_flat = y_pred_binary.flatten()

            precision = precision_score(y_test_flat, y_pred_flat)
            recall = recall_score(y_test_flat, y_pred_flat)
            f1 = f1_score(y_test_flat, y_pred_flat)

            intersection = np.logical_and(y_test_flat, y_pred_flat)
            union = np.logical_or(y_test_flat, y_pred_flat)
            iou = np.sum(intersection) / np.sum(union)

            suffix = f'({fold+1})'
            row[f'F1 - {suffix}'] = round(f1, 4)
            row[f'P - {suffix}'] = round(precision, 4)
            row[f'R - {suffix}'] = round(recall, 4)
            row[f'IoU - {suffix}'] = round(iou, 4)

        results.append(row)

df = pd.DataFrame(results)
columns = ['Model', 'Learning Rate', 'Channels', 'Bands']
for i in range(1, NUM_FOLDS + 1):
    columns += [f'F1 - ({i})', f'P - ({i})', f'R - ({i})', f'IoU - ({i})']

df = df[columns]

df.to_csv('cross_train_results.csv', index=False)
print("Results saved to 'model_results_summary.csv'")