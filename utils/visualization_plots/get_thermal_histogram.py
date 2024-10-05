import glob
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re

import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.layers import *
from keras.models import *
import tifffile as tiff

MAX_PIXEL_VALUE = 65535
CUDA_DEVICE = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

N_CHANNELS = 10
THERMAL_BAND = '10'

PATH_MTL_SCENE = '/media/marycamila/Expansion/raw/2019'
PATH_MTL_VOLCANOES = '/media/marycamila/Expansion/raw/volcanoes'
PATH_MTL_FIRE = '/media/marycamila/Expansion/raw/active_fire/metadata'

# Activefire
WEIGHTS_ACTIVE_FIRE_PATH = '/home/marycamila/flaresat/results_comparison/source/active_fire/model_unet_Voting_10c_final_weights.h5'
THRESHOLD_ACTIVE_FIRE = 0.25

#Flaresat

MODEL_PATH = '/home/marycamila/flaresat/train/train_output/transfer_learning/flaresat-10c-16bs-32f-3lr.hdf5'
THRESHOLD_FLARESAT = 0.50


def open_txt_get_props(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"\s*(\w+)\s*=\s*\"?([^\"]+)\"?", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata


def check_metadata_filepath(file_path):
    if 'flare_patches' in file_path or 'volcano' in file_path:
        entity_id = file_path.split('_')[2]

        if 'flare_patches' in file_path:
            path = os.path.join(PATH_MTL_SCENE, entity_id)
        else:
            path = os.path.join(PATH_MTL_VOLCANOES, entity_id)
                
        metadata_file = glob.glob(os.path.join(path, '*_MTL.txt'))[0]
   
    else:
        # In case of fire patches
        scene_id = file_path.split('/')[6]
        scene_id = scene_id.split('_')[:-1]
        scene_id = '_'.join(scene_id)
        path = os.path.join(PATH_MTL_FIRE, scene_id)
        metadata_file = path + '_MTL.txt'
    
    return metadata_file


def get_patch_kelvin(file_path, tiff_b10, mask):
    img = tiff_b10 * MAX_PIXEL_VALUE  # Scale the TIFF data

    metadata_file = check_metadata_filepath(file_path)    
    metadata = open_txt_get_props(metadata_file)

    # Retrieve thermal constants from metadata
    attribute_mult = 'RADIANCE_MULT_BAND_10'
    attribute_add = 'RADIANCE_ADD_BAND_10'
    attribute_k1 = 'K1_CONSTANT_BAND_10'
    attribute_k2 = 'K2_CONSTANT_BAND_10'

    mult_band = float(metadata[attribute_mult])
    add_band = float(metadata[attribute_add])
    k1_band = float(metadata[attribute_k1])
    k2_band = float(metadata[attribute_k2])

    # Conversion to TOA Radiance
    img = (img * mult_band) + add_band  # Updated to directly assign result to img

    # Conversion to Top of Atmosphere Brightness Temperature
    radiance = np.where(img == 0, 1e-10, img)  # Avoid division by zero
    bt_kelvin = k2_band / np.log((k1_band / radiance) + 1)  # Calculate brightness temperature in Kelvin

    # Transformar a mascara para os quadrados 
    # Pegar as coordenadas para a caixa
    # Pegar o pixel central para preencher o blob

    bt_kelvin_masked = np.where(mask == 1, bt_kelvin, 0)  # Keep the temperature in Kelvin

    return bt_kelvin_masked


def get_mask_patch(file_path):
    mask = tiff.imread(file_path)
    mask = np.resize(mask, (256, 256, 1))
    mask = mask / 255

    return mask


def load_patch(file_path, n_channels, bands=[]):
    img = tiff.imread(file_path)
    img = np.resize(img, (256, 256, 10))
    
    if n_channels == 10:
        img = img[:, :, :]
    else:
        # Active-fire 
        img = img[:, :, bands]
        #img = img[:, :, [1,5,6,4]] # Reference transfer learning
        #img = img[:, :, [1,5,6]] # Reference active-fire
        #img = img[:, :, [4,5,6]] # Reference
        #img = img[:, :, [5,6]]        

    return img


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


def plot_histogram(list_hist, category, method):
    plt.figure(figsize=(10, 6))
    sns.histplot(list_hist, bins=50, color='red', edgecolor='black', kde=True)
    plt.title("Average " + category + " Pixel Temperature Histogram")
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pixel Count')
    plt.grid(True)

    # Save the figure
    plt.savefig(f'{method}_{category}_pixel_temp_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_histogram_all(list_hist_flare, list_hist_fire, list_hist_volcanoes, method):
    plt.figure(figsize=(10, 6))

    # Plot each histogram with different colors and add a label for the legend
    sns.histplot(list_hist_fire, bins=50, color='orange', edgecolor='black', kde=True, label='Fire', stat='density')
    sns.histplot(list_hist_flare, bins=50, color='red', edgecolor='black', kde=True, label='Flare', stat='density')
    sns.histplot(list_hist_volcanoes, bins=50, color='blue', edgecolor='black', kde=True, label='Volcanoes', stat='density')

    # Add the title and labels
    plt.title("Normalized B10 Pixel Temperature Histogram")
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pixel Count')
    plt.grid(True)

    # Add a legend to differentiate the categories
    plt.legend(title="Category")

    # Save the figure
    plt.savefig(f'{method}_pixel_temp_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()


images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')['tiff_file']
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')['mask_file']

hist_avg_flare = []
hist_avg_fire = []
hist_avg_volcanoes = []

hist_pixels_flare = []
hist_pixels_fire = []
hist_pixels_volcanoes = []

# Getting output for flare histogram
active_fire_model = get_unet()
active_fire_model.load_weights(WEIGHTS_ACTIVE_FIRE_PATH)

for index, path in enumerate(images_test):
    tiff_b10 = load_patch(path, N_CHANNELS)[:, :, 8] # Band 10 as we dont have band 8. 
    # tiff_print = cv2.normalize(tiff_b10, None, 0, 255, cv2.NORM_MINMAX)
    # tiff_print = tiff_print.astype(np.uint8)
    # tiff_print = cv2.cvtColor(tiff_print, cv2.COLOR_GRAY2BGR)

    # cv2.imwrite(f'b10_temp_{index}.png', tiff_print)

    if 'flare_patches' in path: 
        mask_celsius = get_mask_patch(masks_test[index])
        mask_celsius = mask_celsius.reshape(256, 256)
    else:       
        patch_full = load_patch(path, N_CHANNELS)
        active_fire_pred = active_fire_model.predict(np.array([patch_full]), batch_size=1)
        mask_celsius = np.where(active_fire_pred[0, :, :, 0] > THRESHOLD_ACTIVE_FIRE, 1, 0)

    mask_patch = get_patch_kelvin(path, tiff_b10, mask_celsius)

    if np.any(mask_patch > 0):
        temp_values_mean = np.mean(mask_patch[mask_patch > 0])

        if 'flare_patches' in path:
            hist_avg_flare.append(temp_values_mean)
        elif 'fire_patches' in path:
            hist_avg_fire.append(temp_values_mean)
        else:
            hist_avg_volcanoes.append(temp_values_mean)

    seg_pixels = mask_patch[mask_patch > 0]
    if seg_pixels.size > 0:
        if 'flare_patches' in path:
            hist_pixels_flare.append(seg_pixels)
        elif 'fire_patches' in path:
            hist_pixels_fire.append(seg_pixels)
        else:
            hist_pixels_volcanoes.append(seg_pixels)


# plot_histogram(hist_avg_flare, 'flare', 'average')
# plot_histogram(hist_avg_fire, 'fire', 'average')
# plot_histogram(hist_avg_volcanoes, 'volcanoes', 'average')

all_flare_pixels = np.concatenate(hist_pixels_flare)
all_fire_pixels = np.concatenate(hist_pixels_fire)
all_volcanoes_pixels = np.concatenate(hist_pixels_volcanoes)
plot_histogram(all_flare_pixels, 'flare', 'global')
plot_histogram(all_fire_pixels, 'fire', 'global')
plot_histogram(all_volcanoes_pixels, 'volcanoes', 'global')

plot_histogram_all(all_flare_pixels, all_fire_pixels, all_volcanoes_pixels, 'full')