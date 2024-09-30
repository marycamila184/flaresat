import numpy as np
import pandas as pd

from utils.metrics import get_metrics_results
from utils.plot_infereces import plot_inferences
from utils.process_scene_toa import *
from methods.comparison_methods import get_toa_texas

OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/output/texas'

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

# Texas gas flare detection reference - https://www.sciencedirect.com/science/article/pii/S1569843222002631
def get_toa_texas_patch(file_path):
    img = get_toa_patch(file_path, 'REFLECTANCE')    
    clear_cloud_mask = get_cloud_mask(file_path)
    flagged_pixels = get_toa_texas(clear_cloud_mask, img)

    return flagged_pixels

test_images = np.array([get_toa_texas_patch(path) for path in images_test['tiff_file']])
test_masks = np.array([get_mask_patch(path) for path in masks_test['mask_file']])
 
y_pred_flat = test_images.flatten()
y_test_flat = test_masks.flatten()

get_metrics_results(y_pred_flat,y_test_flat)
plot_inferences(test_masks, test_images, OUTPUT_PATH, list_entities_plot=[], method="texas", n_images=856)