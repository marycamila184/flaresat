import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import Sequence
import utils.processing as processing

class ImageMaskGenerator(Sequence):
    def __init__(self, n_channels, bands, image_list, mask_list, batch_size, image_size, target_resize=None, shuffle=True):
        self.n_channels = n_channels
        self.bands = bands
        self.image_list = image_list
        self.mask_list = mask_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.target_resize = target_resize
        self.shuffle = shuffle
        self.indexes = np.arange(len(image_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_list[i] for i in batch_indexes]
        batch_masks = [self.mask_list[i] for i in batch_indexes]

        images = np.array([processing.load_image(path, self.n_channels, bands=self.bands, target_size=self.target_resize) for path in batch_images])
        masks = np.array([processing.load_mask(path, target_size=self.target_resize) for path in batch_masks])
        
        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)