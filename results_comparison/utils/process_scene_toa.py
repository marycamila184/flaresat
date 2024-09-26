from skimage.transform import resize
import tifffile as tiff
import numpy as np
import glob
import os
import re

MAX_PIXEL_VALUE = 65535
PATH_MTL_SCENE = '/media/marycamila/Expansion/raw/2019'
PATH_MTL_FIRE = '/media/marycamila/Expansion/raw/active_fire/metadata'
PATH_SCENE_FIRE = '/media/marycamila/Expansion/raw/active_fire/scenes'

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
        path = os.path.join(PATH_MTL_SCENE, entity_id)
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

def get_toa_scene(entity):
    band_list = []

    # Fire scene uses '_' - Example: LC08_L1TP_117016_20200926
    if '_' in entity:
        scene_dir = glob.glob(os.path.join(PATH_SCENE_FIRE, f"*{entity}*"))[0]
    else:
        # Flare scene
        scene_dir = os.path.join(PATH_MTL_SCENE, entity)
        
    metadata_file = glob.glob(os.path.join(scene_dir, '*_MTL.txt'))[0]
    metadata = open_txt_get_props(metadata_file) 

    # Used 6 and 7 bands - Reference: https://www.mdpi.com/2071-1050/15/6/5333
    for band in range(6, 8):
        tiff_path = glob.glob(os.path.join(scene_dir, '*_B' + str(band) + '.TIF'))[0]
        tiff_data = tiff.imread(tiff_path)

        # RXD used Reflectance - Reference: https://www.mdpi.com/2071-1050/15/6/5333 
        attribute_mult = 'REFLECTANCE_MULT_BAND_' + str(band)
        attribute_add = 'REFLECTANCE_ADD_BAND_' + str(band)
        reflectance_mult_band = float(metadata[attribute_mult])
        reflectance_add_band = float(metadata[attribute_add])

        # Conversion to TOA Reflectance - https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
        toa_radiance = (tiff_data * reflectance_mult_band) + reflectance_add_band
        band_list.append(toa_radiance)
    
    toa_scene = np.stack(band_list, axis=-1)

    return toa_scene