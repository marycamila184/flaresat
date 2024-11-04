import glob
import math
import os
from skimage.transform import resize
from rasterio.windows import Window
import tifffile as tiff
import numpy as np
import cv2
import re

PATCH_SIZE = 256

patch = '/home/marycamila/flaresat/dataset/fire_patches/LC08_L1TP_046031_20200908_20200908_01_RT_p00879.tiff'
scene = '/media/marycamila/Expansion/raw/active_fire/scenes/LC08_L1TP_046031_20200908_20200919_02_T1'
patch_index = 879

def get_row_col_from_index(scene_tiff, patch_index):
    rows, cols, _ = scene_tiff.shape

    nx = math.ceil(cols / PATCH_SIZE) 
    ny = math.ceil(rows / PATCH_SIZE) 

    row_patch = int((patch_index - 1) % ny)
    col_patch = int((patch_index - 1) // ny)

    return row_patch, col_patch
    

def get_patch(row_index, col_index, scene_tiff):
    height, width, _= scene_tiff.shape

    row = row_index * 256
    col = col_index * 256

    if (min(PATCH_SIZE, width - col) < PATCH_SIZE):
        col = width - PATCH_SIZE

    if (min(PATCH_SIZE, height - row) < PATCH_SIZE):
        row = height - PATCH_SIZE

    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)

    patch = scene_tiff[window.row_off:window.row_off + window.height,
                 window.col_off:window.col_off + window.width]
    return patch


def open_txt_get_props(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"\s*(\w+)\s*=\s*\"?([^\"]+)\"?", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata


def get_toa_scene(scene_dir, method):
    band_list = []
        
    metadata_file = glob.glob(os.path.join(scene_dir, '*_MTL.txt'))[0]
    metadata = open_txt_get_props(metadata_file) 

    if method == 'RADIANCE':
        len_bands = 10
    else:
        len_bands = 8
    
    for band in range(0, len_bands):
        if band != 7: # Band 8 not used as it has a different resolution
            tiff_path = glob.glob(os.path.join(scene_dir, '*_B' + str(band + 1) + '.TIF'))[0]
            tiff_data = tiff.imread(tiff_path)

            attribute_mult = method + '_MULT_BAND_' + str(band + 1)
            attribute_add = method + '_ADD_BAND_' + str(band + 1)
            mult_band = float(metadata[attribute_mult])
            add_band = float(metadata[attribute_add])

            # Conversion to TOA Reflectance - https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
            toa_scene = (tiff_data * mult_band) + add_band
            band_list.append(toa_scene)
    
    toa_scene = np.stack(band_list, axis=-1)

    return toa_scene


img = tiff.imread(patch)
img = resize(img, (256,256,10))
original_image = img[:, :, 6]
original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)
original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.putText(original_image, 'B7 Landsat', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.imwrite("test_original_patch.png", original_image)

scene_tiff = get_toa_scene(scene, "REFLECTANCE")
row, col = get_row_col_from_index(scene_tiff, patch_index)

patch_tiff = get_patch(row, col, scene_tiff)

cropped = patch_tiff[:, :, 6]
cropped = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX)
cropped = cv2.cvtColor(cropped.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.putText(cropped, 'Cropped B7 Landsat', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.imwrite("cropped_original_patch.png", cropped)


