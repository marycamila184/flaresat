import tensorflow as tf
import pandas as pd
import os

from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

from models.gru_attention_unet import *

from processing import *
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
EPOCHS = 200

IMAGE_SIZE = (256, 256)

N_CHANNELS = 10
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
    mask_list=masks_train,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    n_channels=N_CHANNELS
)

val_generator = ImageMaskGenerator(
    image_list=images_validation,
    mask_list=masks_validation,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    n_channels=N_CHANNELS
)

# DONE - model = unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), base_filters=32)
# DONE - model = attetion_unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), base_filters=32)
# BAD RESULT - model = residual_attention_unet(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), base_filters=32)
# BAD RESULT - model = unet_plus_plus(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), base_filters=32)
# BAD RESULT - model = lstm_unet(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), base_filters=32)
# model = gru_unet(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), base_filters=32)
model = gru_attetion_unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], N_CHANNELS), base_filters=32)

model.summary()

checkpoint = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME),
    monitor='loss',
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