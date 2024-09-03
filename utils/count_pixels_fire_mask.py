import pandas as pd
import numpy as np
import tifffile as tiff
import os

#faço o match com a pasta que as mascaras originais estao 
#conto quantos pixels brancos tem
PATH_DATASET = '/home/marycamila/flaresat/dataset'
PATH_FIRE_MASKS = '/home/marycamila/Downloads/masks_patches'

df_test_fire_mask = pd.read_csv(os.path.join(PATH_DATASET, 'images_fire_mask.csv'))
df_test_fire_mask["mask_file"] = df_test_fire_mask["mask_file"].str.split("/").str[-1]

list_patches = os.listdir(PATH_FIRE_MASKS)
list_test_fire = []

for mask in list_patches:
    mask_parts = mask.split("_")
    filter_mask = mask_parts[0:7] + mask_parts[8:]
    filter_mask = "_".join(filter_mask)
    filter_mask += "f"

    if filter_mask in df_test_fire_mask["mask_file"].values:
        list_test_fire.append(mask)
        
total_fire_pixels = 0
for valid_mask in list_test_fire:
    file_path = os.path.join(PATH_FIRE_MASKS, valid_mask)
    mask = tiff.imread(file_path)
    mask = np.resize(mask, (256, 256, 1))
    count_fire_pixels = np.sum(mask == 1)
    
    total_fire_pixels += count_fire_pixels
    
print("Number of fire pixels in image_fire_mask are: " + str(total_fire_pixels))