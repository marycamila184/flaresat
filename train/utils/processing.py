import numpy as np
import tifffile as tiff
from skimage.transform import resize

IMAGE_SHAPE = (256, 256, 10)
MASK_SHAPE = (256, 256, 1)


def load_image(file_path, n_channels, target_size=None, bands=[]):
    img = tiff.imread(file_path)
    if img.shape != IMAGE_SHAPE:
        raise ValueError(f"{file_path}: expected shape {IMAGE_SHAPE}, got {img.shape}")

    if n_channels == 10:
        img = img[:, :, :]
    else:
        img = img[:, :, bands]
    
    if target_size:
        target_shape = (target_size[0], target_size[1], n_channels)
        img = resize(img, target_shape, preserve_range=True, anti_aliasing=True)

    return img


def load_mask(file_path, target_size=None, norm=True):
    mask = tiff.imread(file_path)
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    if mask.shape != MASK_SHAPE:
        raise ValueError(f"{file_path}: expected shape {MASK_SHAPE}, got {mask.shape}")
    mask = np.float32(mask)

    if target_size:
        target_shape = (target_size[0], target_size[1])
        mask = resize(mask, target_shape, preserve_range=True, anti_aliasing=True)

    if norm:
        mask = np.float32(mask) / 255

    return mask