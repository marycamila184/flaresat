import numpy as np
import pandas as pd
import os

from process_scene_toa import open_txt_get_props
import glob as glob

PATH_MTL_SCENE = '/media/marycamila/Expansion/raw/2019'
PATH_SSD_SCENE = '/media/marycamila/KINGSTON/raw'
PATH_MTL_FIRE = '/media/marycamila/Expansion/raw/active_fire/metadata'

def get_str_entity(file_path):
    if 'flare_patches' in file_path:
        str_entity = file_path.split('_')[2]
    else:
        str_entity = file_path.split('/')[6]
        str_entity = str_entity.split('_')[0:4]
        str_entity = '_'.join(str_entity)

    return str_entity

images = pd.read_csv('/home/marycamila/flaresat/dataset/images_train.csv')
entities_filter = np.array([get_str_entity(path) for path in images['tiff_file']])
unique_entities = list(set(entities_filter))

count_cloud = 0
count_unknown = 0

for entity in unique_entities:
    patches = images[images["tiff_file"].str.contains(entity)]["tiff_file"]

    if '_' in entity:
        scene_dir = glob.glob(os.path.join(PATH_MTL_FIRE, f"*{entity}*"))[0]
        metadata = open_txt_get_props(scene_dir) 
    else:
        try:
            scene_dir = os.path.join(PATH_MTL_SCENE, entity)
            metadata_file = glob.glob(os.path.join(scene_dir, '*_MTL.txt'))[0]
            metadata = open_txt_get_props(metadata_file)
        except:
            scene_dir = os.path.join(PATH_SSD_SCENE, entity)
            metadata_file = glob.glob(os.path.join(scene_dir, '*_MTL.txt'))[0]
            metadata = open_txt_get_props(metadata_file)
    
    cloud_cover = float(metadata['CLOUD_COVER'])
    quality_oli = int(metadata['IMAGE_QUALITY_OLI'])
    quality_tirs = int(metadata['IMAGE_QUALITY_TIRS'])

    if quality_tirs == 9:
        count_cloud += 1
    else:
        print(entity)
        count_unknown += 1

    # if cloud_cover >= 0.00:
    #     count_cloud += 1
    # else:
    #     print("cloud_cover = -1 " + entity)
    #     count_unknown += 1
    

print("Cloud Average: " + str(count_cloud))
print("Unknown cloud: " + str(count_unknown))