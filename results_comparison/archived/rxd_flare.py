import math
import numpy as np
import pandas as pd

from methods.comparison_methods import calculate_global_rxd
from utils.metrics import get_metrics_results
from utils.plot_infereces import plot_inferences
from utils.process_scene_toa import *

from rasterio.windows import Window
import matplotlib.pyplot as plt
import cv2

OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/rxd_output'
THRESHOLD = 0.05 # Reference https://www.mdpi.com/2071-1050/15/6/5333 - 4.2. Anomaly Detection
N_CHANNELS = 2
PATH_MTL = '/media/marycamila/Expansion/raw/2019'
PATH_METADATA = '/media/marycamila/Expansion/raw/active_fire/metadata'
PATCH_SIZE = 256


def get_str_entity(file_path):
    if 'flare_patches' in file_path or 'volcano' in file_path:
        str_entity = file_path.split('_')[2]
    else:
        str_entity = file_path.split('/')[6]
        str_entity = str_entity.split('_')[0:4]
        str_entity = '_'.join(str_entity)

    return str_entity


def find_patch(row_patch, col_patch, tiff_scene):
    height, width = tiff_scene.shape

    row = row_patch * PATCH_SIZE
    col = col_patch * PATCH_SIZE

    if (min(PATCH_SIZE, width - col) < PATCH_SIZE):
        col = width - PATCH_SIZE

    if (min(PATCH_SIZE, height - row) < PATCH_SIZE):
        row = height - PATCH_SIZE

    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)

    patch = tiff_scene[window.row_off:window.row_off + window.height,
                 window.col_off:window.col_off + window.width]
    return patch


images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

test_images = []
test_masks = []
list_entities_plot = []

entities_filter = np.array([get_str_entity(path) for path in images_test['tiff_file']])
unique_entities = list(set(entities_filter))

for entity in unique_entities:
    patches = images_test[images_test["tiff_file"].str.contains(entity)]["tiff_file"]
   
    scene = get_toa_scene(entity)
    rxd_scene_mask = calculate_global_rxd(scene, N_CHANNELS, entity)

    for patch_file in patches:
        if '_' in entity:
            # Entity name conversion to get the patch
            # Example: "'/home/marycamila/flaresat/dataset/fire_patches/LC08_L1TP_025033_20200921_20200921_01_RT_p00410.tiff'"
            patch_index_str = patch_file.split("_")[-1]
            patch_index = int(patch_index_str.replace(".tiff", "").replace("p", ""))

            rows, cols = rxd_scene_mask.shape

            nx = (math.ceil(cols/PATCH_SIZE))
            ny = (math.ceil(rows/PATCH_SIZE))

            row_patch = (patch_index - 1) % ny
            col_patch = (patch_index - 1) // ny

        else:
            row_patch = int(patch_file.split("_")[3])
            col_patch = int(patch_file.split("_")[4])

        rxd_patch_output = find_patch(row_patch, col_patch, rxd_scene_mask)

        test_images.append(rxd_patch_output)
        entity_mask = patch_file.split('/')[-1]

        # RXD scene
        rxd_scene_img = rxd_scene_mask.astype(np.uint8)
        normalized_image = cv2.normalize(rxd_scene_img, None, norm_type=cv2.NORM_MINMAX)
        normalized_image = (normalized_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_PATH, 'test_plot', f'scene_mask_{entity_mask}.png'), normalized_image)

        # RXD patch output
        rxd_patch_img = rxd_patch_output.astype(np.uint8)
        normalized_image = cv2.normalize(rxd_patch_img, None, norm_type=cv2.NORM_MINMAX)
        normalized_image = (normalized_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_PATH, 'test_plot', f'rxd_patch_output{entity_mask}.png'), normalized_image)        

        # Fire mask uses '_' - Example: LC08_L1TP_117016_20200926
        if '_' in entity:
            mask_row = masks_test["mask_file"].str.contains(entity_mask)
        else:
            # Entity name conversion to get the mask
            # Example: "'/home/marycamila/flaresat/dataset/flare_mask_patches/fire_LC81670242019233LGN00_14_27_mask.tiff'"
            entity_mask = entity_mask.replace('patch', 'mask')
            mask_row = masks_test["mask_file"].str.contains(entity_mask)
        
        if len(masks_test[mask_row]["mask_file"]) > 1:
            print("Found more than one mask - Error")
        
        mask_path = masks_test[mask_row]["mask_file"].values[0]
        test_masks.append(get_mask_patch(mask_path))
        list_entities_plot.append(entity_mask)

    print('Entity finished: ' + entity)

y_pred_flat = np.array(test_images).flatten()
y_test_flat = np.array(test_masks).flatten()

get_metrics_results(y_pred_flat,y_test_flat)
plot_inferences(test_masks, test_images, OUTPUT_PATH, list_entities_plot, method="rxd", n_images=150)
