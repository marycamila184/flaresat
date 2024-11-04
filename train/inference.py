from tensorflow.python.keras import backend as K
import utils.processing as processing
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

CUDA_DEVICE = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

IMAGES_NUM = 15
OUTPUT_DIR = '/home/marycamila/flaresat/train/train_output'
MODEL_FILE_NAME = 'flaresat.hdf5'
N_CHANNELS = 10
THRESHOLD = 0.50

def inference_flare(x_test, y_test, y_pred_thresholded, index, save_path):
    """
    Visualize and save the original image, ground truth mask, and predicted mask using OpenCV.
    
    Parameters:
    x_test (numpy array): Array of test images.
    y_test (numpy array): Array of ground truth masks.
    y_pred_thresholded (numpy array): Array of predicted masks.
    index (int): Index of the image to visualize.
    save_path (str): Path to save the visualization.
    """
    # Original image - assuming x_test is a 3D array with shape (n_samples, height, width, n_channels)
    if N_CHANNELS == 10:
        original_image = x_test[index][:, :, 6]  # Display the B7 channel
    elif N_CHANNELS == 3:
        original_image = x_test[index][:, :, 2]  # Display the B7 channel
    original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)
    original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.putText(original_image, 'B7 Landsat', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Ground truth mask - assuming y_test is a 3D array with shape (n_samples, height, width)
    ground_truth_mask = y_test[index].squeeze()
    ground_truth_mask = cv2.normalize(ground_truth_mask, None, 0, 255, cv2.NORM_MINMAX)
    ground_truth_mask = cv2.cvtColor(ground_truth_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.putText(ground_truth_mask, 'Ground Truth Flare Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Predicted mask - assuming y_pred_thresholded is a 3D array with shape (n_samples, height, width)
    predicted_mask = y_pred_thresholded[index].squeeze()
    predicted_mask = cv2.cvtColor(predicted_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.putText(predicted_mask, 'Predicted Flare Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Concatenate images horizontally
    concatenated_image = np.hstack((original_image, ground_truth_mask, predicted_mask))

    # Save the figure
    cv2.imwrite(save_path, concatenated_image)


images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_fire_test.csv').head(IMAGES_NUM)
images_mask = pd.read_csv('/home/marycamila/flaresat/dataset/images_fire_mask.csv').head(IMAGES_NUM)

test_images = np.array([processing.load_image(path, n_channels=N_CHANNELS) for path in images_test['tiff_file']])
mask_images = np.array([processing.load_mask(path) for path in images_mask['mask_file']])

model_path = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
model = tf.keras.models.load_model(model_path)

y_pred = model.predict(test_images)
y_pred_thresholded = np.where(y_pred > THRESHOLD, 1, 0)

bins_y_pred = y_pred.flatten()
y_pred_thresholded = (y_pred_thresholded * 255).astype(np.uint8)

for i in range(IMAGES_NUM):
    file_name = f"inference_{i}.png"
    inference_img = os.path.join(OUTPUT_DIR, file_name)
    inference_flare(test_images, mask_images, y_pred_thresholded, i, inference_img)
