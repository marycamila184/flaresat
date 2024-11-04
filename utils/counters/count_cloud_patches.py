import numpy as np
import glob as glob
import pandas as pd
import os

from metadata_utils import *


PATH_MTL_SCENE = '/media/marycamila/Expansion/raw/2019'
PATH_SSD_SCENE = '/media/marycamila/KINGSTON/raw'
PATH_MTL_FIRE = '/media/marycamila/Expansion/raw/active_fire/metadata'
PATH_MTL_VOLCANO = '/media/marycamila/Expansion/raw/volcanoes'


images = pd.read_csv('/home/marycamila/flaresat/dataset/images_train.csv')
entities_filter = np.array([get_str_entity(path) for path in images['tiff_file']])
unique_entities = list(set(entities_filter))

count_cloud = 0
count_unknown = 0

for entity in unique_entities:
    patches = images[images["tiff_file"].str.contains(entity)]["tiff_file"]

    if '_' in entity:
        # Handle the case where the entity has an underscore
        scene_dir = glob.glob(os.path.join(PATH_MTL_FIRE, f"*{entity}*"))[0]
        metadata = open_txt_get_props(scene_dir)
   
    base_paths = [PATH_MTL_SCENE, PATH_SSD_SCENE, PATH_MTL_VOLCANO]
    scene_dir, metadata_file = get_scene_dir(entity, base_paths)

    if scene_dir and metadata_file:
        metadata = open_txt_get_props(metadata_file)
    
    cloud_cover = float(metadata['CLOUD_COVER'])
    quality_oli = int(metadata['IMAGE_QUALITY_OLI'])
    quality_tirs = int(metadata['IMAGE_QUALITY_TIRS'])

    # if quality_tirs == 9:
    #     count_cloud += 1
    # else:
    #     print(entity)
    #     count_unknown += 1

    if cloud_cover >= 0.00:
        count_cloud += 1
    else:
        print("cloud_cover = -1 " + entity)
        count_unknown += 1
    

print("Cloud Average: " + str(count_cloud))
print("Unknown cloud: " + str(count_unknown))