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

from sklearn.decomposition import PCA
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
PATH_SCENE_FIRE = '/media/marycamila/Expansion/raw/active_fire/scenes'
PATH_MTL_FIRE = '/media/marycamila/Expansion/raw/active_fire/metadata'

# Activefire
WEIGHTS_ACTIVE_FIRE_PATH = '/home/marycamila/flaresat/results_comparison/source/active_fire/model_unet_Voting_10c_final_weights.h5'
THRESHOLD_ACTIVE_FIRE = 0.25

#Flaresat

MODEL_PATH = '/home/marycamila/flaresat/train/train_output/transfer_learning/flaresat-10c-16bs-32f-3lr.hdf5'
THRESHOLD_FLARESAT = 0.50


def get_str_entity(file_path):
    if 'flare_patches' in file_path or 'volcano' in file_path:
        str_entity = file_path.split('_')[2]
    else:
        str_entity = file_path.split('/')[6]
        str_entity = str_entity.split('_')[0:4]
        str_entity = '_'.join(str_entity)

    return str_entity


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


def get_patch_method(file_path, method, mask):
    img = tiff.imread(file_path)
    img = img * MAX_PIXEL_VALUE

    metadata_file = check_metadata_filepath(file_path)    
    metadata = open_txt_get_props(metadata_file)

    if method == 'RADIANCE':
        len_bands = 10
    else:
        len_bands = 8
    
    for band in range(0, len_bands):
        attribute_mult = method + '_MULT_BAND_' + str(band + 1)
        attribute_add = method + '_ADD_BAND_' + str(band + 1)
        mult_band = float(metadata[attribute_mult])
        add_band = float(metadata[attribute_add])

        img[:, :, band] = (img[:, :, band] * mult_band) + add_band

    mask = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
    masked_img = np.where(mask == 1, img, 0)

    return masked_img


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


def plot_scatter(x_values, x_label, y_values, y_label, category, method):
    plt.figure(figsize=(10, 6))    

    x_values = np.concatenate(x_values)
    y_values = np.concatenate(y_values)

    min_length = min(len(x_values), len(y_values))

    x_values = x_values[:min_length]
    y_values = y_values[:min_length]

    sns.scatterplot(x=x_values, y=y_values, s=15)

    plt.title("Scatter Plot of " + category + " - " + x_label + " x " + y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    # Save the figure
    plt.savefig(f'{method}_{category}_{x_label}_{y_label}_pixel_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_scatter(df, category, bands):
    band_lists = [np.concatenate(df[band].tolist()) for band in bands]

    min_length = min(len(band) for band in band_lists)

    trimmed_bands = [band[:min_length] for band in band_lists]
    
    data = np.column_stack(trimmed_bands)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Plot the scatter plot of the principal components
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=10)
    plt.title(f'PCA Scatter Plot for {category}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    plt.savefig(f'pca_{category}_pixel_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    explained_variance = pca.explained_variance_ratio_

    print(f'Explained Variance Ratios for {category}:')
    for i, var in enumerate(explained_variance):
        print(f'PC{i+1}: {var:.2f}')


images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')['tiff_file']
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')['mask_file']

df_pixels_flare = []
df_pixels_fire = []
df_pixels_volcanoes = []

# Getting output for flare histogram
active_fire_model = get_unet()
active_fire_model.load_weights(WEIGHTS_ACTIVE_FIRE_PATH)

for index, path in enumerate(images_test):
    if 'flare_patches' in path: 
        mask_bin = get_mask_patch(masks_test[index])
        mask_bin = mask_bin.reshape(256, 256)
    else:       
        patch_full = load_patch(path, N_CHANNELS)
        active_fire_pred = active_fire_model.predict(np.array([patch_full]), batch_size=1)
        mask_bin = np.where(active_fire_pred[0, :, :, 0] > THRESHOLD_ACTIVE_FIRE, 1, 0)

    mask_patch = get_patch_method(path, 'RADIANCE', mask_bin)
    seg_pixels = mask_patch[mask_patch != 0]

    bands = ['b2', 'b5', 'b6', 'b7']
    band_indices = [1, 4, 5, 6]

    if seg_pixels.size > 0:
        band_pixels = {}

        for band, index in zip(bands, band_indices):
            band_pixels[band] = mask_patch[:, :, index][mask_patch[:, :, index] > 0]

        if 'flare_patches' in path:
            df_pixels_flare.append(band_pixels)
        elif 'fire_patches' in path:
            df_pixels_fire.append(band_pixels)
        else:
            df_pixels_volcanoes.append(band_pixels)

df_pixels_flare = pd.DataFrame(df_pixels_flare)
df_pixels_fire = pd.DataFrame(df_pixels_fire)
df_pixels_volcanoes = pd.DataFrame(df_pixels_volcanoes)

# x_axis = 'b5'
# y_axis = 'b6'

# x_label = 'B5 Band'
# y_label = 'B6 Band'

# plot_scatter(df_pixels_flare[x_axis].tolist(), x_label, df_pixels_flare[y_axis].tolist(), y_label, "flare", "radiance")
# plot_scatter(df_pixels_fire[x_axis].tolist(), x_label, df_pixels_fire[y_axis].tolist(), y_label, "fire", "radiance")
# plot_scatter(df_pixels_volcanoes[x_axis].tolist(), x_label, df_pixels_volcanoes[y_axis].tolist(), y_label, "volcanoes", "radiance")


plot_pca_scatter(df_pixels_flare, "flare", bands)
plot_pca_scatter(df_pixels_fire, "fire", bands)
plot_pca_scatter(df_pixels_volcanoes, "volcanoes", bands)