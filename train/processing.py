import numpy as np
import tifffile as tiff

def get_img_arr(file_path, n_channels):
    img = tiff.imread(file_path)
    img = np.resize(img, (256, 256, 10))
    
    if n_channels == 10:
        img = img[:, :, :]
    elif n_channels == 3:
        # Active-fire 
        img = img[:, :, [1,5,6]]
        #img = img[:, :, [4,5,6]]
    elif n_channels== 2:
        img = img[:, :, [5,6]]

    return img

def get_mask_arr(file_path):
    mask = tiff.imread(file_path)
    mask = np.resize(mask, (256, 256, 1))
    mask = np.float32(mask)/255

    return mask