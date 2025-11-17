import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from models.transfer_learning.unet_attention_sentinel_landcover import ResizeLayer, f1_score

LEARNING_RATE = 0.0005
MASK_CHANNELS = 1


def conv_block(input_tensor, num_filters, kernel_size=3, batch_norm=True, kernel_initializer='he_normal'):
    x = layers.Conv2D(num_filters,kernel_size,padding='same',kernel_initializer=kernel_initializer)(input_tensor)
    if batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_filters,kernel_size,padding='same',kernel_initializer=kernel_initializer)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x) 
    return x


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


def unet_attention_model(input_size, base_filters=32, dropout = 0.1):
    inputs = Input(shape=input_size)
   
    # Downsampling path
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    p1 = layers.Dropout(dropout)(p1)
    
    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    p2 = layers.Dropout(dropout)(p2)
    
    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)
    p3 = layers.Dropout(dropout)(p3)
    
    c4 = conv_block(p3, base_filters * 8)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = layers.Dropout(dropout)(p4)

    c5 = conv_block(p4, base_filters * 16)
   
    # Upsampling path with attention blocks
    u6 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    a6 = attention_block(u6, c4, base_filters * 8)
    u6 = layers.concatenate([u6, a6])
    u6 = layers.Dropout(dropout)(u6)
    c6 = conv_block(u6, base_filters * 8)
    
    u7 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    a7 = attention_block(u7, c3, base_filters * 4)
    u7 = layers.concatenate([u7, a7])
    u7 = layers.Dropout(dropout)(u7)
    c7 = conv_block(u7, base_filters * 4)

    u8 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    a8 = attention_block(u8, c2, base_filters * 2)
    u8 = layers.concatenate([u8, a8])
    u8 = layers.Dropout(dropout)(u8)
    c8 = conv_block(u8, base_filters * 2)

    u9 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    a9 = attention_block(u9, c1, base_filters)
    u9 = layers.concatenate([u9, a9])
    u9 = layers.Dropout(dropout)(u9)
    c9 = conv_block(u9, base_filters)
    
    # Output layer
    outputs = layers.Conv2D(MASK_CHANNELS, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_focal_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score])

    return model