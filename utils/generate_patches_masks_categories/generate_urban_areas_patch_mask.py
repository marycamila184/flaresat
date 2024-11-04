import glob
import tifffile as tiff
from PIL import Image
import pandas as pd
import numpy as np
import os

from rasterio.windows import Window
import rasterio as rio
import tifffile as tiff
import math

import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.layers import *
from keras.models import *

MAX_PIXEL_VALUE = 65535
PATCH_SIZE = 256

CATEGORY = "urban_areas"
ATTR = "urban"

PATH_CSV_POINTS = '/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv'
PATCH_PATH = "/home/marycamila/flaresat/dataset/" + CATEGORY + "_patches"
PATCH_MASK_PATH = "/home/marycamila/flaresat/dataset/" + CATEGORY + "_mask_patches"
PATH_RAW = "/media/marycamila/Expansion/raw/" + CATEGORY
TH_FIRE = 0.25
CUDA_DEVICE = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

df_flare_points = pd.read_csv(PATH_CSV_POINTS)

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


def get_patch_tiff(row, col, tiif_patch):
    height, width, _ = tiif_patch.shape
    row = row * 256
    col = col * 256

    if col + PATCH_SIZE > width:
        col = width - PATCH_SIZE
    if row + PATCH_SIZE > height:
        row = height - PATCH_SIZE

    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
    patch = tiif_patch[window.row_off:window.row_off + window.height,
                       window.col_off:window.col_off + window.width, :]
    return patch


def check_patch_active_fire(patch_tiff):   

    inference_patch = patch_tiff[:,:,[6,5,1]]
    y_pred = model.predict(np.array( [inference_patch] ), batch_size=1)
    result_unet = y_pred[0, :, :, 0] > TH_FIRE    
    num_true_pixels = np.sum(result_unet)

    return num_true_pixels >= 3


def save_tiff_patch_and_mask(patch_img, entity_id, row, col):
    row, col = str(row), str(col)

    patch_filename = f"{ATTR}_{entity_id}_{row}_{col}_patch.tiff"
    patch_file_path = os.path.join(PATCH_PATH, patch_filename)
    tiff.imwrite(patch_file_path, patch_img)

    mask_filename = f"{ATTR}_{entity_id}_{row}_{col}_mask.tiff"
    mask_file_path = os.path.join(PATCH_MASK_PATH, mask_filename)

    # Create mask (currently just a blank mask)
    mask_img = Image.new('L', (256, 256), 0)
    mask_img.save(mask_file_path)


def km_to_deg_lat_lon(lat, distance_km):
    lat_km_per_deg = 111.32
    lon_km_per_deg = 111.32 * math.cos(math.radians(lat))
    
    deg_lat = distance_km / lat_km_per_deg
    deg_lon = distance_km / lon_km_per_deg
    return deg_lat, deg_lon


def read_metadata(entity):

    return None


def is_patch_around_flare_patch(entity, row_index, col_index, patch_size, df_flare_points):
    scene_metadata = read_metadata(entity)
    
    ul_lat = scene_metadata['CORNER_UL_LAT_PRODUCT']
    ul_lon = scene_metadata['CORNER_UL_LON_PRODUCT']
    lr_lat = scene_metadata['CORNER_LR_LAT_PRODUCT']
    lr_lon = scene_metadata['CORNER_LR_LON_PRODUCT']

    total_x_pixels = scene_metadata['REFLECTIVE_SAMPLES']
    total_y_pixels = scene_metadata['REFLECTIVE_LINES']
    lat_per_pixel = (ul_lat - lr_lat) / total_y_pixels
    lon_per_pixel = (lr_lon - ul_lon) / total_x_pixels

    patch_ul_lat = ul_lat - (row_index * patch_size * lat_per_pixel)
    patch_ul_lon = ul_lon + (col_index * patch_size * lon_per_pixel)
    patch_lr_lat = patch_ul_lat - (patch_size * lat_per_pixel)
    patch_lr_lon = patch_ul_lon + (patch_size * lon_per_pixel)

    for _, point in df_flare_points.iterrows():
        lat, lon = point["lat"], point["lng"]

        deg_lat, deg_lon = km_to_deg_lat_lon(lat, 20)
        min_lat = lat - deg_lat
        max_lat = lat + deg_lat
        min_lon = lon - deg_lon
        max_lon = lon + deg_lon

        if not (max_lat < patch_lr_lat or min_lat > patch_ul_lat or max_lon < patch_ul_lon or min_lon > patch_lr_lon):
            return True

    return False
    
list_scenes = os.listdir(PATH_RAW)
list_patch = []

for entity in list_scenes:
    scene_path = os.path.join(PATH_RAW, entity)
    scene_img = preprocessing_tiff(scene_path)

    width, height, _ = scene_img.shape
    for row_index in range(0, height, 256):
        for col_index in range(0, width, 256):
            patch = get_patch_tiff(row_index, col_index)

            if not is_patch_around_flare_patch(entity, patch) and check_patch_active_fire(patch):
                patch_img = np.float32(patch_img) / MAX_PIXEL_VALUE
                list_patch.append({"entity_id_sat": entity, "row_index": row_index, "col_index": col_index})
                save_tiff_patch_and_mask(patch_img, entity, row_index, col_index)
    
df_category_summary = pd.DataFrame(list_patch)
df_category_summary.to_csv("/home/marycamila/flaresat/source/" + CATEGORY + "/patchs_" + CATEGORY + "_fire.csv", index=False)

