from keras.models import Model
from keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.001
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

def gru_unet(input_size, base_filters=32, dropout = 0.1):
    inputs = Input(shape=input_size)
   
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

    # Reshape for GRU
    x = layers.Reshape(target_shape=(c5.shape[1]*c5.shape[2], base_filters * 16))(c5)
    x = layers.GRU(base_filters * 16, return_sequences=False)(x)
    # x = layers.Dense(base_filters * 16, activation='relu')(x)
    x = layers.Reshape(target_shape=c5.shape[1:])(x)

    # Upsampling
    u6 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Dropout(dropout)(u6)
    c6 = conv_block(u6, base_filters * 8)
    
    u7 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(dropout)(u7)
    c7 = conv_block(u7, base_filters * 4)

    u8 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = conv_block(u8, base_filters * 2)

    u9 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(dropout)(u9)
    c9 = conv_block(u9, base_filters)
    
    outputs = layers.Conv2D(MASK_CHANNELS, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_focal_crossentropy', metrics=['accuracy'])

    return model