import tifffile as tiff
from PIL import Image
import pandas as pd
import numpy as np
import os
import re

MAX_PIXEL_VALUE = 65535

PATH_METADATA = '/media/marycamila/Expansion/raw/active_fire/metadata'
PATH_CSV_POINTS = '/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv'
PATCH_ACTIVE_FIRE = '/media/marycamila/Expansion/raw/active_fire/landsat_patches'
PATH_FIRE = '/home/marycamila/flaresat/dataset/fire_patches'
PATH_MASK_FIRE = '/home/marycamila/flaresat/dataset/fire_mask_patches' 

def parse_metadata(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"\s*(\w+)\s*=\s*\"?([^\"]+)\"?", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata


def get_corners(metadata):
    corners = {
        "UL": {"LAT": float(metadata["CORNER_UL_LAT_PRODUCT"]), "LON": float(metadata["CORNER_UL_LON_PRODUCT"])},
        "UR": {"LAT": float(metadata["CORNER_UR_LAT_PRODUCT"]), "LON": float(metadata["CORNER_UR_LON_PRODUCT"])},
        "LL": {"LAT": float(metadata["CORNER_LL_LAT_PRODUCT"]), "LON": float(metadata["CORNER_LL_LON_PRODUCT"])},
        "LR": {"LAT": float(metadata["CORNER_LR_LAT_PRODUCT"]), "LON": float(metadata["CORNER_LR_LON_PRODUCT"])}
    }
    return corners


def overlap_point_corner(lat, lon, corners):
    lat_min = min(corners["UL"]["LAT"], corners["LL"]["LAT"])
    lat_max = max(corners["UR"]["LAT"], corners["LR"]["LAT"])
    lon_min = min(corners["UL"]["LON"], corners["UR"]["LON"])
    lon_max = max(corners["LL"]["LON"], corners["LR"]["LON"])

    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


valid_landsat_scenes = []
list_metadata = os.listdir(PATH_METADATA)
csv_points = pd.read_csv(PATH_CSV_POINTS)

for metadata in list_metadata:
    overlap = False
    path_metadata = os.path.join(PATH_METADATA, metadata)
    metadata = parse_metadata(path_metadata)
    landsat_scene_id = metadata["LANDSAT_PRODUCT_ID"]
    corners = get_corners(metadata)

    for index, point in csv_points.iterrows():
        point_lat = point['Latitude']
        point_lon = point['Longitude']       
        if overlap_point_corner(point_lat, point_lon, corners):
            overlap = True
            break  
    
    if not overlap:
        valid_landsat_scenes.append(landsat_scene_id)

print("Total of valid manually scenes with no overapping with flare images:" + str(len(valid_landsat_scenes)))

## Get the list of entity id to download and for further comparison
# df_test_images = pd.DataFrame(valid_landsat_scenes, columns=["entity_id_active_fire_test"])
# df_test_images.to_csv('entity_id_active_fire_test.csv', index=False)

list_mask_patches = os.listdir('/media/marycamila/Expansion/raw/active_fire/masks_patches')
list_manually_patches = os.listdir('/media/marycamila/Expansion/raw/active_fire/manual_annotations_patches')
list_all_manually_images = list_mask_patches + list_manually_patches

for filename in valid_landsat_scenes:
    list_valid_patches = [img for img in list_all_manually_images if filename in img]

    for valid_patch in list_valid_patches:   
        new_filename = valid_patch.split('_')
        new_filename.pop(7)
        new_filename = '_'.join(new_filename)
        patch_file_path = os.path.join(PATCH_ACTIVE_FIRE, new_filename)
        patch_img = tiff.imread(patch_file_path)
        patch_img = np.resize(patch_img, (256, 256, 10))
        patch_img = np.float32(patch_img) / MAX_PIXEL_VALUE

        new_filename = new_filename.replace('.tif', '.tiff')
        patch_file_path = os.path.join(PATH_FIRE, new_filename)
        tiff.imwrite(patch_file_path, patch_img)

        mask_active_fire = Image.new('L', (256, 256), 0)
        mask_file_path = os.path.join(PATH_MASK_FIRE, new_filename)
        mask_active_fire.save(mask_file_path)