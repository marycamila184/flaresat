import pandas as pd
import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from models.unet import unet_model
from models.attention_unet import unet_attention_model
from models.transfer_learning.unet_sentinel_landcover import unet_sentinel_landcover
from models.transfer_learning.unet_attention_sentinel_landcover import unet_attention_sentinel_landcover

from train.utils.cross_split import create_folds
from train.utils.generator import *

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


EPOCHS = 100
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16

RANDOM_STATE = 42
OUTPUT_DIR = 'train/train_output/cross_validation'

NUM_FOLDS = 4

flare_patches = pd.read_csv('dataset/flare_dataset.csv')
urban_patches = pd.read_csv('dataset/urban_dataset.csv')
wildfire_patches = pd.read_csv('dataset/fire_dataset.csv')

images_flare, images_urban, images_wildfire = create_folds(flare_patches, urban_patches, wildfire_patches)

list_models = ["unet", "unet_attention", "unet_sentinel_landcover", "unet_attention_sentinel_landcover"]
dict_channels = [(1,5,6), (4,5,6), (3,4,5,6), ()]

for model_name in list_models:
    for dict_bands in dict_channels:
        for fold in range(NUM_FOLDS):

            print(f"\n--- Fold {fold + 1} ---")

            train_generator = None
            val_generator = None
            history = None

            # Select fold data - FLARE
            flare_patches = images_flare[images_flare['fold'] != fold]['tiff_file']
            flare_masks = images_flare[images_flare['fold'] != fold]['mask_file']

            # Select fold data - URBAN
            urban_patches = images_urban[images_urban['fold'] != fold]['tiff_file']
            urban_masks = images_urban[images_urban['fold'] != fold]['mask_file']

            # Select fold data - WILDFIRE
            fire_patches = images_wildfire[images_wildfire['fold'] != fold]['tiff_file']
            fire_masks = images_wildfire[images_wildfire['fold'] != fold]['mask_file']

            # Combine for training and validation
            all_patches = pd.concat([flare_patches, urban_patches, fire_patches]).tolist()
            all_masks = pd.concat([flare_masks, urban_masks, fire_masks]).tolist()

            train_patches, val_patches, train_masks, val_masks = train_test_split(
                all_patches,
                all_masks,
                test_size=0.1,
                random_state=42,
                shuffle=True
            )

            n_channels = len(dict_bands) if len(dict_bands) > 0 else 10
            
            # Create generators
            train_generator = ImageMaskGenerator(
                image_list=train_patches,
                bands=list(dict_bands),
                mask_list=train_masks,
                batch_size=BATCH_SIZE,
                image_size=IMAGE_SIZE,
                n_channels=n_channels,
                shuffle=True
            )

            val_generator = ImageMaskGenerator(
                image_list=val_patches,
                bands=list(dict_bands),
                mask_list=val_masks,
                batch_size=BATCH_SIZE,
                image_size=IMAGE_SIZE,
                n_channels=n_channels,
                shuffle=False
            )

            # Build model
            if model_name == "unet":
                model = unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], n_channels))
            elif model_name == "unet_attention":
                model = unet_attention_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], n_channels))
            elif model_name == "unet_sentinel_landcover":
                model = unet_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], n_channels), dict_channels=dict_bands)
            else: # "unet_attention_sentinel_landcover"
                model = unet_attention_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], n_channels), dict_channels=dict_bands)

            band_str = ''.join(str(b) for b in dict_bands) if dict_bands else 'all' if n_channels != 10 else '10'
            checkpoint_model_name = model_name + '_b' + band_str

            checkpoint_path = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}_{checkpoint_model_name}.keras")

            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_f1_score',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )

            # Train model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=EPOCHS,
                callbacks = [checkpoint]
            )

            history_df = pd.DataFrame(history.history)
            history_df['fold'] = fold + 1

            output_history = os.path.join(OUTPUT_DIR, model_name, str(fold + 1))
            history_csv_path = os.path.join(output_history, f"history_fold_{fold+1}_{checkpoint_model_name}.csv")
            history_df.to_csv(history_csv_path, index=False)

            best_epoch = history_df['val_f1_score'].idxmax()
            best_metrics = history_df.loc[best_epoch]
            summary = {
                'fold': fold + 1,
                'best_epoch': best_epoch,
                'val_loss': best_metrics['val_loss'],
                'val_f1': best_metrics['val_f1_score'],        
                'val_precision': best_metrics['val_precision'],
                'val_recall': best_metrics['val_recall']
            }

            summary_df = pd.DataFrame([summary])

            output_summary = os.path.join(OUTPUT_DIR, model_name)
            summary_csv_path = os.path.join(output_summary, f"summary_all_folds_{checkpoint_model_name}.csv")
            summary_df.to_csv(summary_csv_path, mode='a', header=not os.path.exists(summary_csv_path), index=False)


            