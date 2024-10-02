import numpy as np
import pandas as pd
import os

import glob as glob

from metadata_utils import *

PATH_MTL_SCENE = '/media/marycamila/Expansion/raw/2019'
PATH_SSD_SCENE = '/media/marycamila/KINGSTON/raw'
PATH_MTL_FIRE = '/media/marycamila/Expansion/raw/active_fire/metadata'
PATH_MTL_VOLCANO = '/media/marycamila/Expansion/raw/volcanoes'

images = pd.read_csv('/home/marycamila/flaresat/dataset/images_train.csv')

entities_filter = np.array([get_str_entity(path) for path in images['tiff_file']])
unique_entities = list(set(entities_filter))

count_daily = 0
count_night = 0

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
    
    sun_elevatioon = float(metadata['SUN_ELEVATION'])

    if sun_elevatioon > 0:
        count_daily += len(patches)
    else:
        count_night += len(patches)


print("Daily patches: " + str(count_daily))
print("Night patches: " + str(count_night))