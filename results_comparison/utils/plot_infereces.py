import numpy as np
import cv2
import os

def plot_inferences(test_masks, test_images, output_path, method, n_images=30):
    for index in range(n_images):
        original_image = test_masks[index][:, :]
        original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)
        original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(original_image, 'Ground Truth Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        predicted_mask = test_images[index][:, :]
        predicted_mask = predicted_mask.astype(np.uint8)
        predicted_mask = cv2.normalize(predicted_mask, None, 0, 255, cv2.NORM_MINMAX)
        predicted_mask = cv2.cvtColor(predicted_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(predicted_mask, str(method).upper() + ' Predicted Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        concatenated_image = np.hstack((original_image, predicted_mask))

        file_name = f"inference_{method}_{index}.png"
        save_path = os.path.join(output_path, file_name)
        cv2.imwrite(save_path, concatenated_image)