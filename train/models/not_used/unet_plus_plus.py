from keras.layers import Input
from keras.models import Model
from tensorflow.keras import layers

LEARNING_RATE = 0.00001
MASK_CHANNELS = 1

def nested_unet_block(input_tensor, num_filters, kernel_size=3):
    c = layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same')(input_tensor)
    c = layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same')(c)
    return c

def unet_plus_plus(input_size, base_filters):
    inputs = Input(input_size)

    c1 = nested_unet_block(inputs, base_filters)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = nested_unet_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = nested_unet_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = nested_unet_block(p3, base_filters * 8)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = nested_unet_block(p4, base_filters * 16)

    u6 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4, nested_unet_block(u6, base_filters * 8)])
    c6 = nested_unet_block(u6, base_filters * 8)

    u7 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3, nested_unet_block(u7, base_filters * 4)])
    c7 = nested_unet_block(u7, base_filters * 4)

    u8 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2, nested_unet_block(u8, base_filters * 2)])
    c8 = nested_unet_block(u8, base_filters * 2)

    u9 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1, nested_unet_block(u9, base_filters)])
    c9 = nested_unet_block(u9, base_filters)

    outputs = layers.Conv2D(MASK_CHANNELS, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy'])

    return model
