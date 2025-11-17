import os
import numpy as np
import cv2
import tifffile as tiff

from train.utils.processing import load_image

from models.transfer_learning.unet_attention_sentinel_landcover import unet_attention_sentinel_landcover
from models.transfer_learning.unet_sentinel_landcover import unet_sentinel_landcover
from models.attention_unet import unet_attention_model
from models.unet import unet_model


MODEL_PATH = "train/train_output/cross_validation"
OUTPUT_FOLDER = "inference/output"
IMAGE_SIZE = (256, 256)

bands = []  
# Please use: 
# [] for 10 bands 
# [1,5,6] for bands (B2, B6, B7) 
# [4,5,6] for (B5, B6, B7)
# [3,4,5,6] for (B4, B5, B6, B7)

channels = len(bands) or 10

model_name = "unet"  
# Please use: 
# "unet" 
# "unet_attention" 
# "unet_sentinel_landcover" 
# "unet_attention_sentinel_landcover"

fold = 1  
# Used for continental spatial cross training:
# 1 for Asia
# 2 for Africa
# 3 for North America & Oceania
# 4 for Europe & South America


# TIFF input list
list_tiffs = [
    "dataset/flare_patches/fire_LC80020532019213LGN00_21_19_patch.tiff",
    "dataset/flare_patches/fire_LC80020532019245LGN00_22_18_patch.tiff"
]


def save_outputs_cv(image, mask_pred, base_name):
    # Landsat 8 RGB: B4, B3, B2
    rgb = np.stack([image[:, :, 3], image[:, :, 2], image[:, :, 1]], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base_name}_rgb.png"), rgb_bgr)

    # Band 7
    b7 = (image[:, :, 6] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base_name}_b7.png"), b7)

    # Predicted mask
    mask_img = (mask_pred[:, :, 0] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base_name}_pred.png"), mask_img)


def infer_single(model, tiff_path):
    base_name = os.path.basename(tiff_path).replace(".tiff", "")
    image = load_image(file_path=tiff_path, n_channels=channels, bands=dict(bands))

    # Model expects (1, 256, 256, C)
    img_input = np.expand_dims(image, axis=0)

    pred = model.predict(img_input)[0]  # (256,256,1)

    save_outputs_cv(image, pred, base_name)


bands_name = "".join(map(str, bands)) if bands else "10"

# Select model architecture
if model_name == "unet":
    model = unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels))

elif model_name == "unet_attention":
    model = unet_attention_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels))

elif model_name == "unet_sentinel_landcover":
    model = unet_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels),dict_channels=dict(bands))

elif model_name == "unet_attention_sentinel_landcover":
    model = unet_attention_sentinel_landcover(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels),dict_channels=dict(bands))
else:
    raise Exception("Model architecture not found")


weights_path = os.path.join(MODEL_PATH, model_name, f"fold_{fold}", f"fold_{fold}_{model_name}_b{bands_name}.keras")
model.load_weights(weights_path)

for tiff_path in list_tiffs:
    infer_single(model, tiff_path)

print("Inference completed. Generated RGB, B7, and predicted mask in inference/output/.")

