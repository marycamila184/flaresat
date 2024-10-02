import math
from skimage.transform import resize
from rasterio.windows import Window
import tifffile as tiff
import numpy as np

from osgeo import gdal
from osgeo import osr
from osgeo import ogr

gdal.UseExceptions()

import glob
import os
import re

MAX_PIXEL_VALUE = 65535
PATH_MTL_SCENE = '/media/marycamila/Expansion/raw/2019'
PATH_SSD_SCENE = '/media/marycamila/KINGSTON/raw'
PATH_MTL_FIRE = '/media/marycamila/Expansion/raw/active_fire/metadata'
PATH_SCENE_FIRE = '/media/marycamila/Expansion/raw/active_fire/scenes'
PATCH_SIZE = 256

# Reference - https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
# Landsat Collection 2 Level-1 and Level-2 QA Bands


# -------------------- Common methods

def open_txt_get_props(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"\s*(\w+)\s*=\s*\"?([^\"]+)\"?", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata


def get_mask_patch(file_path):
    mask = tiff.imread(file_path)
    mask = np.resize(mask, (256, 256, 1))
    mask = mask / 255

    return mask


# -------------------- PATCH Preprocessing 

def check_metadata_filepath(file_path):
    if 'flare_patches' in file_path:
        entity_id = file_path.split('_')[2]

        try:
            path = os.path.join(PATH_MTL_SCENE, entity_id)
            metadata_file = glob.glob(os.path.join(path, '*_MTL.txt'))[0]     
        except:
            path = os.path.join(PATH_SSD_SCENE, entity_id)
            metadata_file = glob.glob(os.path.join(path, '*_MTL.txt'))[0]    
    else:
        # In case of fire patches
        scene_id = file_path.split('/')[6]
        scene_id = scene_id.split('_')[:-1]
        scene_id = '_'.join(scene_id)
        path = os.path.join(PATH_MTL_FIRE, scene_id)
        metadata_file = path + '_MTL.txt'
    
    return metadata_file


def get_toa_patch(file_path, method):
    img = tiff.imread(file_path)
    img = resize(img, (256,256,10))
    img = img * MAX_PIXEL_VALUE

    metadata_file = check_metadata_filepath(file_path)    
    metadata = open_txt_get_props(metadata_file)

    if method == 'RADIANCE':
        len_bands = 10
    else:
        len_bands = 8
    
    for band in range(0, len_bands):
        if band != 7: # Band 8 not used as it has a different resolution
            
            # Hotspot used Radiance - https://ieeexplore.ieee.org/document/10641298
            # NHI used Radiance - https://ieeexplore.ieee.org/document/9681815
            # Texas used Reflectance - https://www.sciencedirect.com/science/article/pii/S1569843222002631
            attribute_mult = method + '_MULT_BAND_' + str(band + 1)
            attribute_add = method + '_ADD_BAND_' + str(band + 1)
            mult_band = float(metadata[attribute_mult])
            add_band = float(metadata[attribute_add])

            # Conversion to TOA Radiance - https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product    
            img[:, :, band] = (img[:, :, band] * mult_band) + add_band

    return img

# -------------------- SCENE Preprocessing 

def get_row_col(long, lat, path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    target = osr.SpatialReference(wkt=ds.GetProjection())

    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lat, long)

    point.Transform(transform)

    geotransform = ds.GetGeoTransform()

    col = int((point.GetX() - geotransform[0]) / geotransform[1])
    row = int((geotransform[3] - point.GetY()) / abs(geotransform[5]))

    return row, col


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


def get_cloud_mask_patch(row_index, col_index, mask_scene_tiff):
    height, width = mask_scene_tiff.shape

    row = row_index * 256
    col = col_index * 256

    if (min(PATCH_SIZE, width - col) < PATCH_SIZE):
        col = width - PATCH_SIZE

    if (min(PATCH_SIZE, height - row) < PATCH_SIZE):
        row = height - PATCH_SIZE

    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)

    patch = mask_scene_tiff[window.row_off:window.row_off + window.height,
                 window.col_off:window.col_off + window.width]
    return patch


def get_row_col_from_index(scene_cloud_mask, patch_index):
    rows, cols = scene_cloud_mask.shape

    nx = math.ceil(cols / PATCH_SIZE) 
    ny = math.ceil(rows / PATCH_SIZE) 

    row_patch = int((patch_index - 1) % ny)
    col_patch = int((patch_index - 1) // ny)

    return row_patch, col_patch


def get_cloud_mask(file_path):
    if 'flare_patches' in file_path:
        entity_id = file_path.split('_')[2]

        try:
            path = os.path.join(PATH_MTL_SCENE, entity_id)
            qa_band_path = glob.glob(os.path.join(path, '*_QA_PIXEL.TIF'))[0]     
        except:
            path = os.path.join(PATH_SSD_SCENE, entity_id)
            qa_band_path = glob.glob(os.path.join(path, '*_QA_PIXEL.TIF'))[0]
        finally:
            file_parts = file_path.split('_')
            row, col = int(file_parts[3]), int(file_parts[4])
            
            scene_cloud_mask = tiff.imread(qa_band_path)

    else:
        # In case of fire patches
        scene_id = file_path.split('/')[6]
        scene_id = scene_id.split('_')[:-4]
        scene_id = '_'.join(scene_id)
        path = os.path.join(PATH_SCENE_FIRE, scene_id)
        path += '*'
        fire_scene_dir = glob.glob(path)[0]
        qa_band_path = glob.glob(os.path.join(fire_scene_dir, '*_QA_PIXEL.TIF'))[0]
        
        # Entity name conversion to get the patch
        # Example: "'/home/marycamila/flaresat/dataset/non_fire_patches/LC08_L1TP_025033_20200921_20200921_01_RT_p00410.tiff'"
        patch_index_str = file_path.split("_")[-1]
        patch_index = int(patch_index_str.replace(".tiff", "").replace("p", ""))
        
        scene_cloud_mask = tiff.imread(qa_band_path)
        row, col = get_row_col_from_index(scene_cloud_mask, patch_index)

    patch_cloud_mask = get_cloud_mask_patch(row, col, scene_cloud_mask)

    ## Filter for cloud only
    # clouds_bit_mask = 1 << 3
    # cloud_mask = np.where(np.bitwise_and(patch_cloud_mask, clouds_bit_mask) > 0, 1, 0)

    # Filter for cloud and cloud shadow
    clouds_bit_mask = 1 << 3  # Bit 3 for high confidence cloud
    # cloud_shadow_bit_mask = 1 << 4  # Bit 4 for high confidence cloud shadow

    mask = np.bitwise_and(patch_cloud_mask, clouds_bit_mask) > 0
    #| \ (np.bitwise_and(patch_cloud_mask, cloud_shadow_bit_mask) > 0)

    cloud_and_shadow_mask = np.where(mask, 1, 0)

    return cloud_and_shadow_mask