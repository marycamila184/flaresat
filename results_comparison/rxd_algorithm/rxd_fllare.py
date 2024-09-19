import glob
from sklearn.metrics import precision_score, recall_score, f1_score
import tifffile as tiff
import numpy as np
import pandas as pd
import os
import re
import cv2

# Ordenar a lista de imagem e mascara pelo nome da imagem grande
# Filtrar os patches de uma imagem
# Ler a imagem grande
# Fazer os calculos
# For por patch
# Achar o patch da imagem menor
# Extrair o resultado em um vetor com o limiar
# Concatenar na lista
# Flatten 
# Resultado

OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/rxd_algorithm/output'
PATH_MTL = '/media/marycamila/Expansion/raw/2019'
PATH_METADATA = '/media/marycamila/Expansion/raw/active_fire/metadata'
MAX_PIXEL_VALUE = 65535
THRESHOLD = 0.05 # Reference https://www.mdpi.com/2071-1050/15/6/5333 - 4.2. Anomaly Detection

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')


def open_txt_get_props(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"\s*(\w+)\s*=\s*\"?([^\"]+)\"?", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata

def get_toa_img_arr(file_path):
    # NHI flare reference - https://www.mdpi.com/2072-4292/14/24/6319#B29-remotesensing-14-06319
    img = tiff.imread(file_path)
    img = np.resize(img, (256, 256, 10))    
    img = img * MAX_PIXEL_VALUE

    if 'flare_patches' in file_path:
        entity_id = file_path.split('_')[2]
        path = os.path.join(PATH_MTL, entity_id)
        metadata_file = glob.glob(os.path.join(path, '*_MTL.txt'))[0]       
    
    else:
        # In case of fire patches
        scene_id = file_path.split('/')[6]
        scene_id = scene_id.split('_')[:-1]
        scene_id = '_'.join(scene_id)
        path = os.path.join(PATH_METADATA, scene_id)
        metadata_file = path + '_MTL.txt'
    
    metadata = open_txt_get_props(metadata_file)

    # Transform to Conversion to TOA Radiance - https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product    
    for band in range(0, 10):
        if band != 7: # Band 8 not used as it has a different resolution
            attribute_mult = 'RADIANCE_MULT_BAND_' + str(band + 1)
            attribute_add = 'RADIANCE_ADD_BAND_' + str(band + 1)
            radiance_mult_band = float(metadata[attribute_mult])
            radiance_add_band = float(metadata[attribute_add])

            img[:, :, band] = (img[:, :, band] * radiance_mult_band) + radiance_add_band

    img = img[:, :, [5,6]]   

    return img

def get_mask_arr(file_path):
    mask = tiff.imread(file_path)
    mask = np.resize(mask, (256, 256, 1))
    mask = np.float32(mask)/255

    return mask

def calculate_global_rxd():
    image = np.random.rand(7000, 7000, [5,6])  # Example random data

    # Reshape the image to (7000*7000, 10)
    X = image.reshape(-1, 10)

    # Compute the mean vector
    mu = np.mean(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Inverse of the covariance matrix
    cov_matrix_inv = np.linalg.inv(cov_matrix)

    def mahalanobis_distance(x, mu, cov_matrix_inv):
        diff = x - mu
        return np.sqrt(np.dot(np.dot(diff, cov_matrix_inv), diff.T))

    distances = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        distances[i] = mahalanobis_distance(X[i], mu, cov_matrix_inv)

    distances_image = distances.reshape(256, 256)

    anomaly_mask = distances_image > THRESHOLD

    return anomaly_mask

def find_patch(row_patch, col_patch, img):
    patch = []
    return patch

test_images = np.array([get_toa_img_arr(path) for path in images_test['tiff_file']])
test_masks = np.array([get_mask_arr(path) for path in masks_test['mask_file']])

y_pred_flat = test_images.flatten()
y_test_flat = test_masks.flatten()

precision = precision_score(y_test_flat, y_pred_flat)
recall = recall_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat)

intersection = np.logical_and(y_test_flat, y_pred_flat)
union = np.logical_or(y_test_flat, y_pred_flat)
iou = np.sum(intersection) / np.sum(union)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"IoU: {iou}")

for index in range(30):
    original_image = test_masks[index][:, :]
    original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)
    original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.putText(original_image, 'Ground Truth Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    predicted_mask = test_images[index][:, :]
    predicted_mask = cv2.normalize(predicted_mask, None, 0, 255, cv2.NORM_MINMAX)
    predicted_mask = cv2.cvtColor(predicted_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.putText(predicted_mask, 'NHI Predicted Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    concatenated_image = np.hstack((original_image, predicted_mask))

    file_name = f"inference_rxd_{index}.png"
    save_path = os.path.join(OUTPUT_PATH, file_name)
    cv2.imwrite(save_path, concatenated_image)
