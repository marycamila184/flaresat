import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from tensorflow.keras import layers

LEARNING_RATE = 0.00001
MASK_CHANNELS = 1

def residual_block(x, filters, kernel_size=3, padding='same', batch_norm=True):
    conv = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    if batch_norm:
        conv = layers.BatchNormalization()(conv)
    conv = layers.ReLU()(conv)
    
    conv = layers.Conv2D(filters, kernel_size, padding=padding)(conv)
    if batch_norm:
        conv = layers.BatchNormalization()(conv)
    conv = layers.BatchNormalization()(conv)
    
    conv_res = layers.Conv2D(filters, kernel_size=1, padding=padding)(x)
    conv_res = layers.BatchNormalization()(conv_res)
    
    res = layers.Add()([conv_res, conv])
    res = layers.ReLU()(res)
    return x

def attention_block(g, x, num_filters):
    theta_x = layers.Conv2D(num_filters, (1, 1), strides=(1, 1), padding='same')(x)
    phi_g = layers.Conv2D(num_filters, (1, 1), strides=(1, 1), padding='same')(g)

    add = layers.add([theta_x, phi_g])
    relu = layers.Activation('relu')(add)

    psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')(relu)
    sigmoid = layers.Activation('sigmoid')(psi)
    upsampled_sigmoid = tf.image.resize(sigmoid, size=(tf.shape(x)[1], tf.shape(x)[2])) 
    attention = layers.multiply([x, upsampled_sigmoid])
    
    return attention

def residual_attention_unet(input_size, base_filters=32):
    inputs = Input(shape=input_size)
   
    # Downsampling path
    c1 = residual_block(inputs, base_filters)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = residual_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = residual_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = residual_block(p3, base_filters * 8)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = residual_block(p4, base_filters * 16)
   
    # Upsampling path with attention blocks
    u6 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    a6 = attention_block(u6, c4, base_filters * 8)
    u6 = layers.concatenate([u6, a6])
    c6 = residual_block(u6, base_filters * 8)
    
    u7 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    a7 = attention_block(u7, c3, base_filters * 4)
    u7 = layers.concatenate([u7, a7])
    c7 = residual_block(u7, base_filters * 4)

    u8 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    a8 = attention_block(u8, c2, base_filters * 2)
    u8 = layers.concatenate([u8, a8])
    c8 = residual_block(u8, base_filters * 2)

    u9 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    a9 = attention_block(u9, c1, base_filters)
    u9 = layers.concatenate([u9, a9])
    c9 = residual_block(u9, base_filters)
    
    # Output layer
    outputs = layers.Conv2D(MASK_CHANNELS, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy'])

    return model
