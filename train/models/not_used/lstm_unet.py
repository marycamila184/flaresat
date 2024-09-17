from keras.models import Model
from keras.layers import Input
from tensorflow.keras import layers

LEARNING_RATE = 0.00001
MASK_CHANNELS = 1

def conv_block(input_tensor, num_filters, kernel_size=3, dropout_rate=0.1, kernel_initializer='he_normal'):
    x = layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same', kernel_initializer=kernel_initializer)(input_tensor)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same', kernel_initializer=kernel_initializer)(x)
    return x


def lstm_unet(input_size, base_filters=32):
    inputs = Input(shape=input_size)
   
    # Downsampling
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = conv_block(p3, base_filters * 8)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    b1 = conv_block(p4, base_filters * 16)

    x = layers.Reshape(target_shape=(b1.shape[1]*b1.shape[2], base_filters * 16))(b1)
    x = layers.LSTM(base_filters * 16, return_sequences=True)(x)
    x = layers.Reshape(target_shape=b1.shape[1:])(x)

    # Upsampling
    x = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, c4])
    c6 = conv_block(x, base_filters * 8)
    
    u7 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, base_filters * 4)

    u8 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, base_filters * 2)

    u9 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, base_filters)
    
    outputs = layers.Conv2D(MASK_CHANNELS, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy'])

    return model