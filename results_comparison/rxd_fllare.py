import numpy as np
import pandas as pd

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
    if 'flare_patches' in file_path:
        str_entity = file_path.split('_')[2]
    else:
        str_entity = file_path.split('/')[6]
        str_entity = str_entity.split('_')[:-1]
        str_entity = '_'.join(str_entity)

    return str_entity


def calculate_global_rxd(scene):
    X = scene.reshape(-1, N_CHANNELS)
    mu = np.mean(X, axis=0)
    cov_matrix = np.cov(X, rowvar=False)
    cov_matrix_inv = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distances using vectorized operations
    diffs = X - mu
    distances = np.sqrt(np.sum(diffs @ cov_matrix_inv * diffs, axis=1))

    distances_image = distances.reshape(scene.shape[0], scene.shape[1])

    # RXD scene
    #rxd_scene_img = distances_image.astype(np.uint8)
    #rxd_scene_img = (rxd_scene_img * 255).astype(np.uint8)        
    #cv2.imwrite('scene_rxd.png', rxd_scene_img)

    mask_scene = distances_image > THRESHOLD

    return mask_scene


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

list_entities = [get_str_entity(path) for path in images_test['tiff_file']]
unique_entities = list(set(list_entities))

for entity in unique_entities:
    patches = images_test[images_test["tiff_file"].str.contains(entity)]["tiff_file"]
    scene = get_toa_scene(entity)
    rxd_scene_mask = calculate_global_rxd(scene)
    for patch_file in patches:
        row_patch = int(patch_file.split("_")[3])
        col_patch = int(patch_file.split("_")[4])
        rxd_patch_output = find_patch(row_patch, col_patch, rxd_scene_mask)

        # RXD scene
        # rxd_scene_img = rxd_scene_mask.astype(np.uint8)
        # normalized_image = cv2.normalize(rxd_scene_img, None, norm_type=cv2.NORM_MINMAX)
        # normalized_image = (normalized_image * 255).astype(np.uint8)
        # cv2.imwrite('scene_mask.png', normalized_image)

        # RXD patch output
        # rxd_patch_img = rxd_patch_output.astype(np.uint8)
        # normalized_image = cv2.normalize(rxd_patch_img, None, norm_type=cv2.NORM_MINMAX)
        # normalized_image = (normalized_image * 255).astype(np.uint8)
        # cv2.imwrite('rxd_patch_output.png', normalized_image)

        test_images.append(rxd_patch_output)

        if 'LC08_L1TP' in entity:
            mask_row = masks_test["mask_file"].str.contains(f'{entity}.tiff')
        else:
            mask_row = masks_test["mask_file"].str.contains(f'{entity}_{row_patch}_{col_patch}_mask.tiff')
        
        if len(masks_test[mask_row]["mask_file"]) > 1:
            print("Found more than one mask - Error")
        
        mask_path = masks_test[mask_row]["mask_file"].values[0]
        test_masks.append(get_mask_patch(mask_path))

y_pred_flat = test_images.flatten()
y_test_flat = test_masks.flatten()

get_metrics_results(y_pred_flat,y_test_flat)
plot_inferences(test_masks, test_images, OUTPUT_PATH, method="rxd")
