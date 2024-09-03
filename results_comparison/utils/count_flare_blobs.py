from scipy.ndimage import label
import tifffile as tiff
import pandas as pd
import numpy as np
import os

PATH_DATASET = '/home/marycamila/flaresat/dataset'

df_test_mask = pd.read_csv(os.path.join(PATH_DATASET, 'masks_val.csv'))["mask_file"]

total_blob_count = 0

for patch in df_test_mask:
    mask = tiff.imread(patch)
    mask = np.resize(mask, (256, 256, 1))
    labeled_mask, num_blobs = label(mask == 255)
    total_blob_count += num_blobs

    # print(f"File: {patch}, Blobs: {num_blobs}")

print(f"Total number of white blobs: {total_blob_count}")
    