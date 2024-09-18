import numpy as np
import tifffile as tiff
from skimage.transform import resize

def load_image(file_path, n_channels, target_size=None, bands=[]):
    img = tiff.imread(file_path)
    img = np.resize(img, (256, 256, 10))
    
    if n_channels == 10:
        img = img[:, :, :]
    elif n_channels == 3 or n_channels == 4:
        # Active-fire 
        img = img[:, :, bands]
        #img = img[:, :, [1,5,6,4]] # Refernce transfer learning
        #img = img[:, :, [1,5,6]] # Reference active-fire
        #img = img[:, :, [4,5,6]] # Refernce
    elif n_channels== 2:
        img = img[:, :, [5,6]]
    
    if target_size:
        target_shape = (target_size[0], target_size[1], n_channels)
        img = resize(img, target_shape, preserve_range=True, anti_aliasing=True)

    return img

def load_mask(file_path, target_size=None, norm=True):
    mask = tiff.imread(file_path)
    mask = np.resize(mask, (256, 256, 1))
    mask = np.float32(mask)

    if target_size:
        target_shape = (target_size[0], target_size[1])
        mask = resize(mask, target_shape, preserve_range=True, anti_aliasing=True)

    if norm:
        mask = np.float32(mask) / 255

    return mask