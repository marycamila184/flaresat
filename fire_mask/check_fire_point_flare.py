import shutil
import os
import pandas as pd
import numpy as np
from PIL import Image
import tifffile as tiff
import cv2

YEAR = "2019"
MONTH = "03"

# Constants
PATH_MASK = "/home/marycamila/flaresat/dataset/flare_mask_patches"
PATH_SQUARE = "/home/marycamila/flaresat/dataset/flare_patches_square"
PATH_FLARE = "/home/marycamila/flaresat/dataset/flare_patches"

PATH_FIRE = "/home/marycamila/flaresat/fire_mask/fire_images/" + YEAR + "/" + MONTH
PATH_CSV = "/home/marycamila/flaresat/source/landsat_scenes/2019/active_fire"
PATH_PATCHES = "/home/marycamila/flaresat/source/landsat_scenes/2019/valid_patches"
PATCH_SIZE = 256

# 750 meter from VIIRs precision used to generate flare points
# https://www.mdpi.com/1996-1073/9/1/14
HALF_SIZE_SQUARE = 35

def summary_white_blobs(df_: pd.DataFrame, entity: str) -> list:
    list_fire_detected = []

    for row_index in df_['row_index'].unique():
        df_grouped = df_[df_["row_index"] == row_index]

        for col_index in df_grouped['col_index'].unique():
            prefix = f"fire_{entity}_{row_index}_{col_index}"
            path = os.path.join(PATH_FIRE, f"{prefix}_mask.tiff")

            image = tiff.imread(path)
            image = np.resize(image, (256, 256, 1))
            image_pil = Image.fromarray(image[:, :, 0])
            
            # Thresholding
            binary_image = np.array(image_pil.point(lambda p: 255 if p > 200 else 0))

            # Convert to a format suitable for OpenCV
            binary_image = (binary_image > 0).astype(np.uint8) * 255

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

            # Extract the centroids of each blob
            for i in range(1, num_labels):
                centroid = centroids[i]
                size = stats[i, cv2.CC_STAT_AREA]
                list_fire_detected.append({
                    "entity_id_sat": entity,
                    "row_index": row_index,
                    "col_index": col_index,
                    "row_blob": int(centroid[1]),
                    "col_blob": int(centroid[0]),
                    "size": size
                })

    return list_fire_detected


def check_overlapping_blob(df_filter_white: pd.DataFrame, df_filter_fire: pd.DataFrame) -> pd.DataFrame:
    for row_index in df_filter_white['row_index'].unique():
        df_grouped = df_filter_white[df_filter_white["row_index"] == row_index]

        for col_index in df_grouped['col_index'].unique():
            df_f_f = df_filter_white[(df_filter_white["row_index"] == row_index) & (df_filter_white["col_index"] == col_index)]
            df_p_f = df_filter_fire[(df_filter_fire["row_index"] == row_index) & (df_filter_fire["col_index"] == col_index)]

            for index_blob, item_blob in df_f_f.iterrows():
                count = 0
                row_blob, col_blob = item_blob["row_blob"], item_blob["col_blob"]

                for _, item_point in df_p_f.iterrows():
                    col = item_point["col"] - (PATCH_SIZE * col_index)
                    row = item_point["row"] - (PATCH_SIZE * row_index)

                    area = int((item_point["point_area"] * 2) / 30)

                    if not (row + (HALF_SIZE_SQUARE + area) <= row_blob or
                            row - (HALF_SIZE_SQUARE + area) >= row_blob or
                            col + (HALF_SIZE_SQUARE + area) <= col_blob or
                            col - (HALF_SIZE_SQUARE + area) >= col_blob):
                        count += 1

                df_filter_white.loc[index_blob, "squares_overlapping"] = count

    return df_filter_white


def move_file(file_name, source_dir, dest_dir):
    source_path = os.path.join(source_dir, file_name)
    dest_path = os.path.join(dest_dir, file_name)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        print(f'File copied: {file_name}')
    else:
        print(f'File not found: {file_name}')


if __name__ == "__main__":
    csv_path = os.path.join(PATH_CSV, f"scenes_{MONTH}_queue.csv")
    df = pd.read_csv(csv_path)
    
    df_fire = df[df["pixels_count_fire"] > 0]
    list_scenes = df_fire["entity_id_sat"].unique()
    
    white_blobs = []
    
    for entity in list_scenes:
        df_filtered = df_fire[df_fire["entity_id_sat"] == entity]
        list_white_blob_summary = summary_white_blobs(df_filtered, entity)
        white_blobs.append(pd.DataFrame(list_white_blob_summary))
    
    df_white_blobs = pd.concat(white_blobs)

    
    df_white_blobs["squares_overlapping"] = 0
    list_summary = []

    for entity in list_scenes:
        df_filter_white = df_white_blobs[df_white_blobs["entity_id_sat"] == entity]
        df_filter_fire = df_fire[df_fire["entity_id_sat"] == entity]

        df_blob_validated = check_overlapping_blob(df_filter_white, df_filter_fire)
        list_summary.append(pd.DataFrame(df_blob_validated))

    df_flare_blobs = pd.concat(list_summary)
    
    grouped = df_flare_blobs.groupby(['entity_id_sat', 'row_index', 'col_index'])
    count_flare = 0
    colunt_invalid_flare = 0

    list_valid_flares = []
    list_test = []

    for name, group in grouped:
        if (group['squares_overlapping'] == 0).any():
            colunt_invalid_flare += 1
        else:
            for index, patch in group.iterrows():
                file_path_prefix = "fire_" + patch["entity_id_sat"] + "_" + str(patch["row_index"]) + "_" + str(patch["col_index"])
                count_flare += 1
                list_valid_flares.append(group)
                list_test.append(file_path_prefix)

                # Move patch
                patch_file = file_path_prefix + "_patch.tiff"
                move_file(patch_file, PATH_FIRE, PATH_FLARE)

                # Move mask
                mask_file = file_path_prefix + "_mask.tiff"
                move_file(mask_file, PATH_FIRE, PATH_MASK)

                # Move square
                square_file = file_path_prefix + "_square.tif"
                move_file(square_file, PATH_FIRE, PATH_SQUARE)

                print(f'File copied {file_path_prefix}')

    print("Unique patches with validated flares: " + str(len(set(list_test))))
    df_valid_flares = pd.concat(list_valid_flares)
    file_name = os.path.join(PATH_PATCHES, f"valid_patches_{MONTH}.csv")
    
    df_valid_flares.to_csv(file_name, index=False)
            