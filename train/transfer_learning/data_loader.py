from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import torch

class ImageGenerator(Dataset):
    def __init__(self, train_images, train_masks, num_channels, num_classes, image_shape):
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.train_images = train_images
        self.train_masks = train_masks

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        img = tiff.imread(self.train_images[idx])
        img = np.resize(img, (self.image_shape[0], self.image_shape[1], self.num_channels))
        
        if self.num_channels == 10:
            img = img[:, :, :]
        elif self.num_channels == 3:
            # Active-fire 
            img = img[:, :, [1,5,6]]
            #img = img[:, :, [4,5,6]]
        elif self.num_channels == 2:
            img = img[:, :, [5,6]]

        mask = tiff.imread(self.train_masks[idx])
        mask = np.resize(mask, (256, 256, 1))
        mask = np.float32(mask)/255

        # Adjusted for tensor input
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long).squeeze(2)
        
        return img, mask
