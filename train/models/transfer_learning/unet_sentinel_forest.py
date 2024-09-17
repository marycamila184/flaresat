import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np

LEARNING_RATE = 0.0001
MASK_CHANNELS = 1

def unet_sentinel_forest(input_size):
    model_path = '/home/marycamila/flaresat/train/models/transfer_learning/models/unet-sentinel-forest-4c.hdf5'
    base_model = tf.keras.models.load_model(model_path)
    
    base_output = base_model.get_layer('conv2d_843').output
    
    x = MaxPooling2D(pool_size=(2, 2))(base_output)
    new_output = Conv2D(MASK_CHANNELS, (1, 1), activation='sigmoid', name='new_output')(x)
    
    model = Model(inputs=base_model.input, outputs=new_output)
    
    model.summary()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_focal_crossentropy", metrics=['accuracy'])
    
    return model

# Call the function with the new input size (512, 512, 10)
unet_sentinel_forest(input_size=(256, 256, 10))
