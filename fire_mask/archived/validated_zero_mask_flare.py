import math
import shutil
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import tifffile as tiff

YEAR = "2019"
MONTH = "08"

# Constants
PATH_MASK = "/home/marycamila/flaresat/dataset/flare_mask_patches"
PATH_SQUARE = "/home/marycamila/flaresat/dataset/flare_patches_square"
PATH_FLARE = "/home/marycamila/flaresat/dataset/flare_patches"

PATH_FIRE = "/home/marycamila/flaresat/fire_mask/fire_images/" + YEAR + "/" + MONTH
PATH_CSV = "/home/marycamila/flaresat/source/landsat_scenes/2019/active_fire"
PATH_PATCHES = "/home/marycamila/flaresat/source/landsat_scenes/2019/valid_patches"
PATCH_SIZE = 256

def move_file(file_name, source_dir, dest_dir):
    source_path = os.path.join(source_dir, file_name)
    dest_path = os.path.join(dest_dir, file_name)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        print(f'File copied: {file_name}')
    else:
        print(f'File not found: {file_name}')


def read_mask(entity, row_index, col_index):
    prefix = f"fire_{entity}_{row_index}_{col_index}"
    path = os.path.join(PATH_FIRE, f"{prefix}_mask.tiff")

    image = tiff.imread(path)
    image = np.resize(image, (256, 256, 1))
    image_pil = Image.fromarray(image[:, :, 0])

    binary_image = np.array(image_pil.point(lambda p: 255 if p > 200 else 0))

    return binary_image


if __name__ == "__main__":
    csv_path = os.path.join(PATH_CSV, f"scenes_{MONTH}_queue.csv")
    df = pd.read_csv(csv_path)
    
    df_fire = df[df["pixels_count_fire"] > 0]
    list_scenes = df_fire["entity_id_sat"].unique()
    
    white_blobs = []

    for entity in list_scenes:
        df_filtered = df_fire[df_fire["entity_id_sat"] == entity]
        list_rows = df_filtered["row_index"].unique()
        list_cols = df_filtered["col_index"].unique()

        for row_index in list_rows:
            for col_index in list_cols:
                df_row_col = df_filtered[(df_filtered["row_index"] == row_index) & (df_filtered["col_index"] == col_index)]
                if len(df_row_col) > 0:
                    mask_tiff = read_mask(entity, row_index, col_index)
                    mask_zero = np.zeros_like(mask_tiff)

                    binary_image = (mask_tiff > 0).astype(np.uint8) * 255
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

                    for _, item_point in df_row_col.iterrows():
                        area = int((math.sqrt(item_point["point_ellip"] * 1_000_000) / 2) / 30)
                        mask_filter = np.zeros_like(mask_tiff)
                        col = item_point["col"] - (PATCH_SIZE * col_index)
                        row = item_point["row"] - (PATCH_SIZE * row_index)

                        row_start = max(0, row - area)
                        row_end = min(mask_tiff.shape[0], row + area)  
                        col_start = max(0, col - area)
                        col_end = min(mask_tiff.shape[1], col + area)

                        for i in range(1, num_labels):
                            centroid = centroids[i]
                            size = stats[i, cv2.CC_STAT_AREA]
                            row_blob = int(centroid[1])
                            col_blob = int(centroid[0])

                            if row_start <= row_blob < row_end and col_start <= col_blob < col_end:
                                mask_filter[row_start:row_end, col_start:col_end] = 1
                            
                            mask_zero |= mask_filter
                        
                    mask_tiff = np.where(mask_zero, mask_tiff, 0)
                    
                    file_path_prefix = "fire_" + entity + "_" + str(row_index) + "_" + str(col_index)

                    patch_file = file_path_prefix + "_patch.tiff"
                    move_file(patch_file, PATH_FIRE, PATH_FLARE)

                    tiff.imwrite(PATH_MASK + "/" + file_path_prefix + '_mask.tiff', mask_tiff)

                    tiff.imwrite(PATH_SQUARE + "/" + file_path_prefix + '_mask.tiff', (mask_zero * 255))