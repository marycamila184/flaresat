import tifffile as tiff
import pandas as pd
import numpy as np
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import cv2 
import re
import os
import glob as glob
from rasterio.windows import Window
import math
import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.layers import *
from keras.models import *


PATH_CSV_POINTS = '/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv'
PATCH_ACTIVE_FIRE = '/media/marycamila/Expansion/raw/active_fire/masks_patches'
PATCH_SIZE = 256
MODEL_PATH = '/home/marycamila/flaresat/train/train_output/transfer_learning/flaresat-10c-16bs-32f-3lr.hdf5'
WEIGHTS_ACTIVE_FIRE_PATH = '/home/marycamila/flaresat/results_comparison/source/active_fire/model_unet_Voting_10c_final_weights.h5'
THRESHOLD_ACTIVE_FIRE = 0.25
THRESHOLD_FLARESAT = 0.50
MAX_PIXEL_VALUE = 65535
N_CHANNELS = 10

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


def get_unet(input_height=256, input_width=256, n_filters=64, dropout=0.1, batchnorm=True):
    input_img = Input(shape=(input_height, input_width, N_CHANNELS))

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


def get_row_col_flare(long, lat, path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    target = osr.SpatialReference(wkt=ds.GetProjection())

    width = ds.RasterXSize

    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lat, long)

    point.Transform(transform)

    geotransform = ds.GetGeoTransform()

    # Calculate the column (X) and row (Y) indices
    col = int((point.GetX() - geotransform[0]) / geotransform[1])
    row = int((geotransform[3] - point.GetY()) / abs(geotransform[5]))

    return row, col


def get_corners(metadata):
    corners = {
        "UL": {"LAT": float(metadata["CORNER_UL_LAT_PRODUCT"]), "LON": float(metadata["CORNER_UL_LON_PRODUCT"])},
        "UR": {"LAT": float(metadata["CORNER_UR_LAT_PRODUCT"]), "LON": float(metadata["CORNER_UR_LON_PRODUCT"])},
        "LL": {"LAT": float(metadata["CORNER_LL_LAT_PRODUCT"]), "LON": float(metadata["CORNER_LL_LON_PRODUCT"])},
        "LR": {"LAT": float(metadata["CORNER_LR_LAT_PRODUCT"]), "LON": float(metadata["CORNER_LR_LON_PRODUCT"])}
    }
    return corners


def get_patch(row_index, col_index, scene_tiff):
    height, width, _= scene_tiff.shape

    row = row_index * 256
    col = col_index * 256

    if (min(PATCH_SIZE, width - col) < PATCH_SIZE):
        col = width - PATCH_SIZE

    if (min(PATCH_SIZE, height - row) < PATCH_SIZE):
        row = height - PATCH_SIZE

    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)

    patch = scene_tiff[window.row_off:window.row_off + window.height,
                 window.col_off:window.col_off + window.width, :]
    return patch


def open_txt_get_props(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"\s*(\w+)\s*=\s*\"?([^\"]+)\"?", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata


def get_toa_scene(scene_dir):
    band_list = []
        
    metadata_file = glob.glob(os.path.join(scene_dir, '*_MTL.txt'))[0]
    metadata = open_txt_get_props(metadata_file) 
    
    for band in range(0, 11):
        if band != 7: # Band 8 not used as it has a different resolution
            tiff_path = glob.glob(os.path.join(scene_dir, '*_B' + str(band + 1) + '.TIF'))[0]
            tiff_data = tiff.imread(tiff_path)
            band_list.append(tiff_data)
    
    toa_scene = np.stack(band_list, axis=-1)

    return toa_scene, metadata


def overlap_point_corner(lat, lon, corners):
    lat_min = min(corners["UL"]["LAT"], corners["LL"]["LAT"])
    lat_max = max(corners["UR"]["LAT"], corners["LR"]["LAT"])
    lon_min = min(corners["UL"]["LON"], corners["UR"]["LON"])
    lon_max = max(corners["LL"]["LON"], corners["LR"]["LON"])

    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def get_row_col_from_index(scene_tiff, patch_index):
    rows, cols, _ = scene_tiff.shape

    nx = math.ceil(cols / PATCH_SIZE) 
    ny = math.ceil(rows / PATCH_SIZE) 

    row_patch = int((patch_index - 1) % ny)
    col_patch = int((patch_index - 1) // ny)

    return row_patch, col_patch
    

csv_points = pd.read_csv(PATH_CSV_POINTS)
scene, metadata = get_toa_scene("/home/marycamila/Downloads/LC08_L1TP_193029_20200914_20200919_02_T1")

landsat_scene_id = metadata["LANDSAT_PRODUCT_ID"]
corners = get_corners(metadata)

list_points_flare = []

for index, point in csv_points.iterrows():
    point_lat = point['Latitude']
    point_lon = point['Longitude']       
    if overlap_point_corner(point_lat, point_lon, corners):
        list_points_flare.append({"latitude": point_lat, "longitude": point_lon})
    
scene_name = "LC08_L1TP_193029_20200914_20200914_01_RT_*"

list_files = glob.glob(os.path.join(PATCH_ACTIVE_FIRE, scene_name))

list_patches = []

for file_name in list_files:
    patch_number = file_name.split("_")[-1]
    patch_number = patch_number.replace(".tif", "").replace("p", "")
    patch_number= int(patch_number)

    list_patches.append(patch_number)

list_patches = set(list_patches)
# Somente um patch com flare

path_b1 = "/home/marycamila/Downloads/LC08_L1TP_193029_20200914_20200919_02_T1/LC08_L1TP_193029_20200914_20200919_02_T1_B1.TIF"
latitude = list_points_flare[0]['latitude']
longitude = list_points_flare[0]['longitude']

row_flare, col_flare = get_row_col_flare(longitude, latitude, path_b1)

print(len(list_patches))
model = tf.keras.models.load_model(MODEL_PATH)

active_fire_model = get_unet()
active_fire_model.load_weights(WEIGHTS_ACTIVE_FIRE_PATH)

list_patch_flare = []

for patch_number in list_patches:
    row, col = get_row_col_from_index(scene, patch_number)

    if row != row_flare and col != col_flare:
        patch_tif = get_patch(row, col, scene)
        patch_tif = np.float32(patch_tif)/MAX_PIXEL_VALUE
        patch_tif = np.expand_dims(patch_tif, axis=0)

        y_pred = model.predict(patch_tif)
        y_pred_thresholded = np.where(y_pred > THRESHOLD_FLARESAT, 1, 0)

        method_masks = active_fire_model.predict(patch_tif)
        method_masks_binary = np.where(method_masks > THRESHOLD_ACTIVE_FIRE, 1, 0)

        if np.any(y_pred_thresholded) or np.any(method_masks_binary):
            print("Patch with Fire or Flare")    
            list_patch_flare.append(patch_number)

            original_image = patch_tif[0, :, :, 6]
            original_image = original_image.reshape(256, 256, 1)            
            original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)
            original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.putText(original_image, 'B7 Landsat', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            method_mask = method_masks_binary.reshape(256, 256, 1)
            method_mask = cv2.normalize(method_mask, None, 0, 255, cv2.NORM_MINMAX)
            method_mask = cv2.cvtColor(method_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.putText(method_mask, 'Active Fire Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            flaresat_mask = y_pred_thresholded.reshape(256, 256, 1)
            flaresat_mask = cv2.normalize(flaresat_mask, None, 0, 255, cv2.NORM_MINMAX)
            flaresat_mask = cv2.cvtColor(flaresat_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.putText(flaresat_mask, 'Predicted Flare Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            masks_to_concatenate = [original_image, method_mask, flaresat_mask]
            concatenated_image = np.hstack(masks_to_concatenate)

            file_name = f"../output/patches_urban/inference_urbean_areas_{patch_number}.png"
            cv2.imwrite(file_name, concatenated_image)
    
print(len(list_patch_flare))