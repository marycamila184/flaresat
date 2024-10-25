import tensorflow as tf
import pandas as pd
import os

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt

from models.transfer_learning.unet_sentinel_landcover import unet_sentinel_landcover
from models.attention_unet import unet_attention_model
from models.unet import unet_model

from generator import *

CUDA_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

# Check GPU availability and configure memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

CHECKPOINT_MODEL_NAME = 'flaresat.hdf5'
EPOCHS = 150

IMAGE_SIZE = (256, 256)

N_CHANNELS = 3
BANDS = [3,5,6]
DICT_CHANNELS = (3,5,6)
BATCH_SIZE = 16

RANDOM_STATE = 42
OUTPUT_DIR = '/home/marycamila/flaresat/train/train_output'

images_train = pd.read_csv('/home/marycamila/flaresat/dataset/images_train.csv')['tiff_file']
masks_train = pd.read_csv('/home/marycamila/flaresat/dataset/masks_train.csv')['mask_file']
images_validation = pd.read_csv('/home/marycamila/flaresat/dataset/images_val.csv')['tiff_file']
masks_validation = pd.read_csv('/home/marycamila/flaresat/dataset/masks_val.csv')['mask_file']

# Create instances of the custom data generator
train_generator = ImageMaskGenerator(
    image_list=images_train,
    bands=BANDS,
    mask_list=masks_train,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    n_channels=N_CHANNELS,
    augment=True 
)

val_generator = ImageMaskGenerator(
    image_list=images_validation,
    bands = BANDS,
    mask_list=masks_validation,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    n_channels=N_CHANNELS
)

#model = unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS))
model = unet_attention_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS))
#model = unet_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), add_attention=False, dict_channels=DICT_CHANNELS)
#model = unet_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), add_attention=True, dict_channels=DICT_CHANNELS)
model.summary()

model_view = os.path.join(OUTPUT_DIR, "model_architecture.png")
plot_model(model, to_file=model_view, show_shapes=True, show_layer_names=True)

checkpoint = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto',
    save_freq='epoch'
)

print('FlareSat - Train initiated')

history = model.fit(
        train_generator,
        validation_data=val_generator,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint]
)

print('FlareSat - Train finished!')

#Plot the model training history
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"), dpi=300, bbox_inches='tight')
plt.clf()
    
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"), dpi=300, bbox_inches='tight')
plt.clf()