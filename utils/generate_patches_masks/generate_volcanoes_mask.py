import glob
import tifffile as tiff
from PIL import Image
import pandas as pd
import numpy as np
import os

from rasterio.windows import Window
import rasterio as rio
import tifffile as tiff

MAX_PIXEL_VALUE = 65535
PATCH_SIZE = 256

PATH_CSV_POINTS = '/home/marycamila/flaresat/source/volcanoes/scenes_points_volcanoes.csv'
PATCH_VOLCANO = '/home/marycamila/flaresat/dataset/volcanoes_patches'
PATCH_MASK_VOLCANO = '/home/marycamila/flaresat/dataset/volcanoes_mask_patches'
PATH_RAW = '/media/marycamila/Expansion/raw/volcanoes'
df = pd.read_csv(PATH_CSV_POINTS)


def preprocessing_tiff(fpath):
    band_list = []

    for band in range (1, 12):
        if band != 8:
            path = glob.glob(os.path.join(fpath, '*_B' + str(band) + '.TIF'))[0]
            tiff = rio.open(path).read(1)
            band_list.append(tiff)

    tiff = np.transpose(band_list, (1, 2, 0))

    return tiff


def get_patch_tiff(row_index, col_index, tiff_volcano):
    height, width, _ = tiff_volcano.shape

    row = row_index * 256
    col = col_index * 256

    if (min(PATCH_SIZE, width - col) < PATCH_SIZE):
        col = width - PATCH_SIZE

    if (min(PATCH_SIZE, height - row) < PATCH_SIZE):
        row = height - PATCH_SIZE

    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)

    patch = tiff_volcano[window.row_off:window.row_off + window.height,
                 window.col_off:window.col_off + window.width, :]
    return patch


for row in df.itertuples(): 
    entity_id = row.entity_id_sat
    row_index = row.row_index
    col_index = row.col_index

    scene_path = os.path.join(PATH_RAW, entity_id)

    scene_img = preprocessing_tiff(scene_path)
    patch_img = get_patch_tiff(row_index, col_index, scene_img)
    patch_img = np.float32(patch_img) / MAX_PIXEL_VALUE

    row_index, col_index = str(row_index), str(col_index)
    
    filename = "volcano_" + entity_id + "_" +row_index+ "_" +col_index+ "_patch.tiff"
    patch_file_path = os.path.join(PATCH_VOLCANO, filename)    
    tiff.imwrite(patch_file_path, patch_img)

    filename = "volcano_" + entity_id + "_" +row_index+ "_" +col_index+ "_mask.tiff"
    mask_volcano = Image.new('L', (256, 256), 0)
    mask_file_path = os.path.join(PATCH_MASK_VOLCANO, filename)
    mask_volcano.save(mask_file_path)