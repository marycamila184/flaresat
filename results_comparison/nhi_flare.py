import numpy as np
import pandas as pd

from methods.comparison_methods import get_toa_nhi
from utils.metrics import get_metrics_results
from utils.plot_infereces import plot_inferences
from utils.process_scene_toa import *

OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/output/nhi'
PATH_MTL = '/media/marycamila/Expansion/raw/2019'
PATH_METADATA = '/media/marycamila/Expansion/raw/active_fire/metadata'
MAX_PIXEL_VALUE = 65535

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

# NHI flare reference - https://ieeexplore.ieee.org/document/9681815
def get_toa_nhi_patch(file_path):
    img = get_toa_patch(file_path, 'RADIANCE')    
    hp = get_toa_nhi_patch(img)
    return hp

test_images = np.array([get_toa_nhi(path) for path in images_test['tiff_file']])
test_masks = np.array([get_mask_patch(path) for path in masks_test['mask_file']])

y_pred_flat = test_images.flatten()
y_test_flat = test_masks.flatten()

get_metrics_results(y_pred_flat,y_test_flat)
plot_inferences(test_masks, test_images, OUTPUT_PATH, list_entities_plot=[], method="nhi", n_images=50)