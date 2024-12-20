import numpy as np
from models.transfer_learning.builder import unet_builder
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer
import tensorflow as tf

LEARNING_RATE = 0.001
MASK_CHANNELS = 1

class ResizeLayer(Layer):
    def __init__(self):
        super(ResizeLayer, self).__init__()

    def call(self, inputs):
        size = (tf.shape(inputs)[1], tf.shape(inputs)[2])
        return tf.image.resize(inputs, size=size)
    

def attention_block(g, x, num_filters):
    resize_layer = ResizeLayer()
    
    theta_x = layers.Conv2D(num_filters, (1, 1), strides=(1, 1), padding='same')(x)
    phi_g = layers.Conv2D(num_filters, (1, 1), strides=(1, 1), padding='same')(g)

    add = layers.add([theta_x, phi_g])
    relu = layers.Activation('relu')(add)

    psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')(relu)
    sigmoid = layers.Activation('sigmoid')(psi)
    
    upsampled_sigmoid = resize_layer(sigmoid)
    attention = layers.multiply([x, upsampled_sigmoid])
    return attention


def unet_attention_sentinel_landcover(input_size, dict_channels=None, seed=42):
    n_channels = input_size[2]
    new_model = unet_builder.build_unet(n_channels, MASK_CHANNELS, activation='sigmoid')  # Flare binary output

    existing_model = unet_builder.build_unet(14, 13, activation='softmax')  # Original model - 13 classes
    existing_model.load_weights('/home/marycamila/flaresat/train/models/transfer_learning/models/unet-sentinel-landcover-14c.h5')
    existing_weights = existing_model.get_weights()

    # Mapping Sentinel-2 bands to Landsat 8 bands
    band_2 = existing_weights[0][:, :, 1, :]  # Sentinel Band 2 → Landsat Band 2 (Blue)
    band_3 = existing_weights[0][:, :, 2, :]  # Sentinel Band 3 → Landsat Band 3 (Green)
    band_4 = existing_weights[0][:, :, 3, :]  # Sentinel Band 4 → Landsat Band 4 (Red)
    band_5 = existing_weights[0][:, :, 4, :]  # Sentinel Band 8 → Landsat Band 5 (NIR)
    band_6 = existing_weights[0][:, :, 7, :]  # Sentinel Band 11 → Landsat Band 6 (SWIR1)
    band_7 = existing_weights[0][:, :, 10, :]  # Sentinel Band 12 → Landsat Band 7 (SWIR2)

    input_shape = existing_weights[0].shape

    band_1 = HeNormal(seed=seed)((input_shape[0], input_shape[1], input_shape[3])).numpy()  # Landsat Band 1 (Coastal Aerosol)
    band_9 = HeNormal(seed=seed+2)((input_shape[0], input_shape[1], input_shape[3])).numpy()  # Landsat Band 9 (Cirrus)
    band_10 = HeNormal(seed=seed+3)((input_shape[0], input_shape[1], input_shape[3])).numpy()  # Landsat Band 10 (Thermal Infrared 1)
    band_11 = HeNormal(seed=seed+4)((input_shape[0], input_shape[1], input_shape[3])).numpy()  # Landsat Band 11 (Thermal Infrared 2)
    
    if n_channels == 3:            
        if dict_channels == (1,5,6): # 2, 6 e 7 Landsat OLI            
            list_channels_weights = [band_2, band_6, band_7]

        elif dict_channels == (4,5,6): # 5, 6 e 7 Landsat OLI
            list_channels_weights = [band_5, band_6, band_7]   

        elif dict_channels == (3,5,6): # 4, 6 e 7 Landsat OLI
            list_channels_weights = [band_4, band_6, band_7]  

    elif n_channels == 4:
        # 4, 5, 6 e 7 Landsat OLI
        list_channels_weights = [band_4, band_5, band_6, band_7]  
        
    else:
        list_channels_weights = [band_1, band_2, band_3, band_4, band_5, band_6, band_7, band_9, band_10, band_11]
        
    existing_weights[0] = existing_weights[0][:, :, :n_channels, :]
    existing_weights[0][:, :, :n_channels, :] = np.stack(list_channels_weights, axis=2)

    shape_weights = existing_weights[-2].shape
    new_output_weights_shape = (shape_weights[0], shape_weights[1], shape_weights[2], 1)  
    existing_weights[-2] = np.zeros(new_output_weights_shape) 

    new_output_bias_shape = (1,)
    existing_weights[-1] = np.zeros(new_output_bias_shape)

    new_model.set_weights(existing_weights)

    inputs = new_model.input
    encoder_outputs = {
        'decoder_stage0_relu1': new_model.get_layer('encoder_stage3_relu2').output,
        'decoder_stage1_relu1': new_model.get_layer('encoder_stage2_relu2').output,
        'decoder_stage2_relu1': new_model.get_layer('encoder_stage1_relu2').output,
        'decoder_stage3_relu1': new_model.get_layer('encoder_stage0_relu2').output
    }

    x = inputs

    for layer in new_model.layers:        
        if 'concatenate' not in layer.name and 'input_layer' not in layer.name:
            if layer.name in encoder_outputs:            
                skip_connection = encoder_outputs[layer.name]
                attended_skip = attention_block(layer.output, skip_connection, num_filters=layer.output.shape[-1])
                x = layer(x)
                x = layers.Concatenate()([attended_skip, x])
            else:
                x = layer(x)

    new_model = Model(inputs=inputs, outputs=x)
    new_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_focal_crossentropy', metrics=['accuracy'])

    return new_model