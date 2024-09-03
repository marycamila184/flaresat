import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
import tifffile as tiff

class ImageMaskGenerator(Sequence):
    def __init__(self, n_channels, image_list, mask_list, batch_size, image_size, shuffle=True):
        self.n_channels = n_channels
        self.image_list = image_list
        self.mask_list = mask_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(image_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_list[i] for i in batch_indexes]
        batch_masks = [self.mask_list[i] for i in batch_indexes]
        
        images = np.array([self.load_image(path, self.n_channels) for path in batch_images])
        masks = np.array([self.load_mask(path) for path in batch_masks])
        
        return images, masks

    def load_image(self, file_path, n_channels):
        img = tiff.imread(file_path)
        img = np.resize(img, (256, 256, 10))
        
        if n_channels == 10:
            return img
        elif n_channels == 3:
            # Active-fire 
            return img[:, :, [1, 5, 6]]
            #return img[:, :, [4, 5, 6]]
        elif n_channels== 2:
            return img[:, :, [5, 6]]
        else:
            raise ValueError("Unsupported number of channels")

    def load_mask(self, file_path):
        mask = tiff.imread(file_path)
        mask = np.resize(mask, (256, 256, 1))
        mask = np.float32(mask) / 255
        return mask

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)