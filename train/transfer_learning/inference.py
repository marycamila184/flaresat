import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os

THRESHOLD = 0.50
IMAGE_SIZE = (256, 256)
N_CHANNELS = 10

class InferenceModel:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def predict(self):
        self.model.eval()
        self.predictions = []
        for img_path in self.images_paths:
            img = self.preprocess_image(img_path).to(self.device)
            with torch.no_grad():
                pred = self.model(img)
                pred = torch.sigmoid(pred).cpu().numpy().squeeze()
                pred_binary = (pred > self.threshold).astype(np.uint8) * 255
                self.predictions.append(pred_binary)

    def visualize_and_save(self, index, save_path):
        original_image = self.preprocess_image(self.images_paths[index]).cpu().numpy().squeeze()
        ground_truth_mask = self.preprocess_mask(self.masks_paths[index])

        if self.n_channels == 10:
            original_image = original_image[6, :, :] 
        elif self.n_channels == 3:
            original_image = original_image[2, :, :] 
        original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)
        original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(original_image, 'B7 Landsat', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        ground_truth_mask = cv2.normalize(ground_truth_mask, None, 0, 255, cv2.NORM_MINMAX)
        ground_truth_mask = cv2.cvtColor(ground_truth_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(ground_truth_mask, 'Ground Truth Flare Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        predicted_mask = cv2.cvtColor(self.predictions[index], cv2.COLOR_GRAY2BGR)
        cv2.putText(predicted_mask, 'Predicted Flare Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        concatenated_image = np.hstack((original_image, ground_truth_mask, predicted_mask))
        cv2.imwrite(save_path, concatenated_image)


model_path = '/home/marycamila/flaresat/train/train_output/flare-sentinel.pth'
model = torch.load(model_path)

inference_model = InferenceModel(model, device='cuda', threshold=0.5)
inference_model.load_images('/path/to/images_test.csv', '/path/to/masks_test.csv', n_channels=10)
inference_model.predict()

for i in range(15):
    file_name = f"inference_{i}.png"
    save_path = os.path.join('/path/to/output_dir', file_name)
    inference_model.visualize_and_save(i, save_path)
