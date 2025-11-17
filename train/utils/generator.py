import random
import numpy as np
from tensorflow.keras.utils import Sequence
import utils.processing as processing

random.seed(42)

class ImageMaskGenerator(Sequence):
    def __init__(self, n_channels, bands, image_list, mask_list, batch_size, image_size, target_resize=None, shuffle=True, augment=False):
        self.n_channels = n_channels
        self.bands = bands
        self.image_list = image_list
        self.mask_list = mask_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.target_resize = target_resize
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(image_list))
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))


    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_list[i] for i in batch_indexes]
        batch_masks = [self.mask_list[i] for i in batch_indexes]

        images = []
        masks = []

        for img_path, mask_path in zip(batch_images, batch_masks):
            image = self.process_image(img_path)
            mask = self.process_mask(mask_path)
            
            # Apply synchronized augmentation to both image and mask
            if self.augment:
                image, mask = self.apply_augmentation(image, mask)
            
            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)
    

    def process_image(self, path):
        return processing.load_image(path, self.n_channels, bands=self.bands, target_size=self.target_resize)


    def process_mask(self, path):
        return processing.load_mask(path, target_size=self.target_resize)


    def apply_augmentation(self, image, mask):
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        if random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        return image, mask


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)