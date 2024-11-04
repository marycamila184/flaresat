from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

LEARNING_RATE = 0.001
MASK_CHANNELS = 1

def unet_sentinel_forest(input_size):
    #model_path = '/home/marycamila/flaresat/train/models/transfer_learning/models/unet-attention-4d.hdf5'
    model_path = '/home/marycamila/flaresat/train/models/transfer_learning/models/unet-attention-3d.hdf5'
    #model_path = '/home/marycamila/flaresat/train/models/transfer_learning/models/unet-attention-4d-atlantic.hdf5'
    model = load_model(model_path)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_focal_crossentropy", metrics=['accuracy'])

    return model
