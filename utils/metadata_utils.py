import glob
import os
import re

def open_txt_get_props(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"\s*(\w+)\s*=\s*\"?([^\"]+)\"?", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata


def get_str_entity(file_path):
    if 'flare_patches' in file_path or 'volcano' in file_path:
        str_entity = file_path.split('_')[2]
    else:
        str_entity = file_path.split('/')[6]
        str_entity = str_entity.split('_')[0:4]
        str_entity = '_'.join(str_entity)

    return str_entity


def get_scene_dir(entity, base_paths):
    """Try to find the scene directory in the provided base paths."""
    for base_path in base_paths:
        scene_dir = os.path.join(base_path, entity)
        metadata_files = glob.glob(os.path.join(scene_dir, '*_MTL.txt'))
        if metadata_files:
            return scene_dir, metadata_files[0]
    return None, None