import glob
import math
import tifffile as tiff
from PIL import Image
import pandas as pd
import numpy as np
import os

from rasterio.windows import Window
import rasterio as rio
import tifffile as tiff

import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.layers import *
from keras.models import *

MAX_PIXEL_VALUE = 65535
PATCH_SIZE = 256

PATH_CSV_POINTS = '/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv'
PATCH_PATH = "/home/marycamila/flaresat/dataset/urban_areas_patches"
PATCH_MASK_PATH = "/home/marycamila/flaresat/dataset/urban_areas_mask_patches"
PATH_RAW = "/media/marycamila/Expansion/raw/urban_areas"
TH_FIRE = 0.25
CUDA_DEVICE = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_height=256, input_width=256, n_filters=16, dropout=0.1, batchnorm=True, n_channels=3):
    input_img = Input(shape=(input_height, input_width, n_channels))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2,
                      kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4,
                      kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8,
                      kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16,
                      kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8,
                      kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4,
                      kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2,
                      kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1,
                      kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


df = pd.read_csv("/home/marycamila/flaresat/source/urban_areas/scenes_points_urban_areas_queue.csv")

weights_path = "/home/marycamila/flaresat/fire_mask/model_unet_Voting_3c_final_weights.h5"
model = get_unet()
model.load_weights(weights_path)


def preprocessing_tiff(fpath):
    band_list = []

    for band in range (1, 12):
        if band != 8:
            path = glob.glob(os.path.join(fpath, '*_B' + str(band) + '.TIF'))[0]
            tiff = rio.open(path).read(1)
            band_list.append(tiff)

    tiff = np.transpose(band_list, (1, 2, 0))

    return tiff


def get_patches_tiff(row, col, tiif_patch):
    height, width, _ = tiif_patch.shape
    patch_size = PATCH_SIZE * 2  # 512x512 patch (4 sub-patches of 256x256)

    if col + patch_size > width:
        col = width - patch_size
    if row + patch_size > height:
        row = height - patch_size

    window = Window(col, row, patch_size, patch_size)
    patch = tiif_patch[window.row_off:window.row_off + window.height,
                       window.col_off:window.col_off + window.width, :]

    half_patch_size = PATCH_SIZE  # 256 if PATCH_SIZE is 256
    patch_list = [
        patch[0:half_patch_size, 0:half_patch_size, :],         # Top-left
        patch[0:half_patch_size, half_patch_size:patch_size, :], # Top-right
        patch[half_patch_size:patch_size, 0:half_patch_size, :], # Bottom-left
        patch[half_patch_size:patch_size, half_patch_size:patch_size, :]  # Bottom-right
    ]

    return patch_list


def check_patch_active_fire(patch_tiff):   
    inference_patch = patch_tiff[:,:,[6,5,1]]
    y_pred = model.predict(np.array( [inference_patch] ), batch_size=1)
    result_unet = y_pred[0, :, :, 0] > TH_FIRE    
    num_true_pixels = np.sum(result_unet)

    return num_true_pixels >= 2


def save_tiff_patch_and_mask(patch_img, entity_id, city, index):
    index = str(index)

    patch_filename = f"urban_{entity_id}_{city}_{index}_patch.tiff"
    patch_file_path = os.path.join(PATCH_PATH, patch_filename)
    tiff.imwrite(patch_file_path, patch_img)

    mask_filename = f"urban_{entity_id}_{city}_{index}_mask.tiff"
    mask_file_path = os.path.join(PATCH_MASK_PATH, mask_filename)

    # Create mask
    mask_img = Image.new('L', (256, 256), 0)
    mask_img.save(mask_file_path)

df_city = pd.read_csv("/home/marycamila/flaresat/source/urban_areas/points_urban_areas_valid.csv")
df_city = df_city[["city","lat","lng"]]

df = df.merge(df_city, left_on=["urban_latitude", "urban_longitude"], right_on=["lat","lng"])

list_patch = []

for row in df.itertuples():
    entity_id = row.entity_id_sat
    city = row.city
    coord_row = row.row
    coord_col = row.col

    scene_path = os.path.join(PATH_RAW, entity_id)
    scene_img = preprocessing_tiff(scene_path)

    patch_img_list = get_patches_tiff(coord_row, coord_col, scene_img)
    patch_img_list = np.float32(patch_img_list) / MAX_PIXEL_VALUE

    for index_patch, patch_img in enumerate(patch_img_list):
        if check_patch_active_fire(patch_img):
            list_patch.append({"entity_id_sat": entity_id, "city": city, "index_patch": index_patch})
            save_tiff_patch_and_mask(patch_img, entity_id, city, index_patch)
    
df_category_summary = pd.DataFrame(list_patch)
df_category_summary.to_csv("/home/marycamila/flaresat/source/urban_areas/patchs_urban_areas_fire.csv", index=False)

