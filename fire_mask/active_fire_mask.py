import glob
import os
from osgeo import gdal
import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.layers import *
from keras.models import *
import numpy as np
from rasterio.windows import Window
import cv2
import rasterio as rio
import tifffile as tiff
import pandas as pd

gdal.UseExceptions()

YEAR = "2019"
MONTH = "03"

TH_FIRE = 0.25
PATH_FIRE = "/home/marycamila/flaresat/fire_mask/fire_images/" + YEAR + "/" + MONTH
PATH_RAW = "/media/marycamila/Expansion/raw"
PATH_CSV = "/home/marycamila/flaresat/source/landsat_scenes"
CUDA_DEVICE = 0
PATCH_SIZE = 256
MAX_PIXEL_VALUE = 65535
CHANNELS = 10

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass


def preprocessing_tiff(fpath):
    band_list = []

    for band in range (1, 12):
        if band != 8:
            path = glob.glob(os.path.join(fpath, '*_B' + str(band) + '.TIF'))[0]
            tiff = rio.open(path).read(1)
            band_list.append(tiff)

    tiff = np.transpose(band_list, (1, 2, 0))

    return tiff


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


def get_unet(input_height=256, input_width=256, n_filters=64, dropout=0.1, batchnorm=True, n_channels=CHANNELS):
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


def get_patch_tiff(row_index, col_index, tiff):
    height, width, _ = tiff.shape

    row = row_index * 256
    col = col_index * 256

    if (min(PATCH_SIZE, width - col) < PATCH_SIZE):
        col = width - PATCH_SIZE

    if (min(PATCH_SIZE, height - row) < PATCH_SIZE):
        row = height - PATCH_SIZE

    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)

    patch = tiff[window.row_off:window.row_off + window.height,
                 window.col_off:window.col_off + window.width, :]
    return patch


def create_queue_csv():
    # Setting the original dataset to use q queue
    df_points = pd.read_csv(PATH_CSV + "/" + YEAR + "/active_fire/scenes_" + MONTH + "_queue.csv")
    df_points["fire_processed"] = False
    df_points["pixels_count_fire"] = -1
    # Setting a queue to stop the processing in case it is needed
    df_points.to_csv(PATH_CSV + "/" + YEAR + "/active_fire/scenes_" + MONTH + "_queue.csv", index=False)


def main():
    # Load model
    weights_path = "/home/marycamila/flaresat/fire_mask/model_unet_Voting_" +str(CHANNELS)+ "c_final_weights.h5"
    model = get_unet()
    model.load_weights(weights_path)

    # Getting already processed points
    df = pd.read_csv(PATH_CSV + "/" + YEAR + "/active_fire/scenes_" + MONTH + "_queue.csv")

    # Pegando somente os pontos validos.
    df_points = df[~df["fire_processed"]]

    list_images = df_points["entity_id_sat"].unique()

    print(" ------ Start Processing Entities ------ ")

    for image_sat in list_images:
        print("Initiated - Entity Id:" + image_sat)

        df_entity = df_points[df_points["entity_id_sat"] == image_sat]

        root_path_image = os.path.join(PATH_RAW, YEAR, image_sat)

        multichannel_tiff = preprocessing_tiff(root_path_image)
        img = np.float32(multichannel_tiff)/MAX_PIXEL_VALUE

        for entity in df_entity.itertuples():
            patch_img = get_patch_tiff(entity.row_index, entity.col_index, img)

            if CHANNELS == 3:
                inference_patch = patch_img[:,:,[6, 5, 1]]
            else:
                inference_patch = patch_img

            y_pred = model.predict(np.array( [inference_patch] ), batch_size=1)
            result_unet = y_pred[0, :, :, 0] > TH_FIRE

            num_true_pixels = np.sum(result_unet)

            prefix = "fire_" + str(image_sat) + "_" + str(entity.row_index) + "_" + str(entity.col_index)

            if num_true_pixels >= 3:
                print("Fire detected for Patch:" + prefix)
                print("Pixels count: " +str(num_true_pixels))

                patch_tiff = patch_img.reshape(256, 256, 10)
                tiff.imwrite( PATH_FIRE + "/" + prefix + '_patch.tiff', patch_tiff)

                # Print RGB patch
                # red = patch_img[:, :, 3]
                # green = patch_img[:, :, 2]
                # blue = patch_img[:, :, 1]

                # rgb_image = np.stack([blue, green, red ], axis=-1)
                # rgb_image = (rgb_image * 255).astype(np.uint8)

                # try:
                #     cv2.imwrite( PATH_FIRE + "/" + prefix + '_rgb.png', rgb_image)
                # except Exception as e:
                #     print(f"Error while writing the image: {e}")

                b7_image = patch_img[:, :, 6]
                b7_image = (b7_image * 255).astype(np.uint8)

                cv2.imwrite( PATH_FIRE + "/" + prefix + '_B7.png', b7_image)
                
                # Print mask
                mask_unet = result_unet.reshape(256, 256, 1)
                mask_unet = (mask_unet * 255).astype(np.uint8)

                mask_unet[mask_unet > TH_FIRE] = 255
                tiff.imwrite(PATH_FIRE + "/" + prefix + '_mask.tiff', mask_unet)
                
                df.at[entity.Index,'fire_processed'] = True
                df.at[entity.Index,'pixels_count_fire'] = num_true_pixels
            else:

                print("No fire detected for row:" + str(entity.row_index) +" col:" + str(entity.col_index) +" - Entity Id:" + image_sat)
                df.at[entity.Index, 'fire_processed'] = True

        df.to_csv(PATH_CSV + "/" + YEAR + "/active_fire/scenes_" + MONTH + "_queue.csv", index=False)
        print("Finished - Entity Id:" + image_sat)


if __name__ == "__main__":
    #create_queue_csv()
    main()
