import numpy as np
from keras.models import Model
from keras.layers import Input
from models.transfer_learning.builder import unet_builder
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.001
MASK_CHANNELS = 1

def unet_sentinel_landcover(input_size):
    n_channels = input_size[2]
    new_model = unet_builder.build_unet(n_channels, MASK_CHANNELS, activation='sigmoid')  # Flare binary output

    # https://github.com/mayrajeo/lulc_ml
    # https://jyx.jyu.fi/handle/123456789/60705
    # Land cover classification from multispectral data using convolutional autoencoder networks
    existing_model = unet_builder.build_unet(14, 13, activation='softmax')  # Original model - 13 classes
    existing_model.load_weights('/home/marycamila/flaresat/train/models/transfer_learning/models/unet-sentinel-landcover-14c.h5')
    existing_weights = existing_model.get_weights()

    existing_weights[0] = existing_weights[0][:, :, :n_channels, :]

    shape_weights = existing_weights[-2].shape
    new_output_weights_shape = (shape_weights[0], shape_weights[1], shape_weights[2], 1)  
    existing_weights[-2] = np.zeros(new_output_weights_shape) 

    new_output_bias_shape = (1,)
    existing_weights[-1] = np.zeros(new_output_bias_shape)

    new_model.set_weights(existing_weights)

    for layer in new_model.layers:
        if "encoder" in layer.name:
            layer.trainable = False
        
        print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

    new_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_focal_crossentropy', metrics=['accuracy'])

    return new_model