import numpy as np
import pandas as pd

from utils.metrics import get_metrics_results
from utils.plot_infereces import plot_inferences
from utils.process_scene_toa import *

OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/output/texas'
PATH_MTL = '/media/marycamila/Expansion/raw/2019'
PATH_METADATA = '/media/marycamila/Expansion/raw/active_fire/metadata'
MAX_PIXEL_VALUE = 65535

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

# Texas gas flare detection reference - https://www.sciencedirect.com/science/article/pii/S1569843222002631
def get_toa_texas_patch(file_path):
    img = get_toa_patch(file_path, 'REFLECTANCE')    
    img = img[:, :, [4,5,6]]   

    # Reference https://www.sciencedirect.com/science/article/pii/S1569843222002631 and https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
    pnir = img [:, :, 0]
    pnear_swir = img [:, :, 1]
    pfar_swear = img [:, :, 2]
    
    # TAI equation
    tai = (pfar_swear-pnear_swir) / pnir

    # First filter
    flaring_pixels = np.where((tai >= 0.15) & (pfar_swear >= 0.15), 1, 0)

    # In case of NO
    # saturated_pixels = (pfar_swear > 1) & (pnear_swir > pfar_swear)
    # unambigous_fire_pixels = np.where(flaring_pixels | saturated_pixels, 1, 0)
    
    # In case of Yes
    second_filter = (pnear_swir > 0.05) & (pnir > 0.01)

    # Merged all filter results
    flagged_pixels = np.where(second_filter & flaring_pixels, 1, 0)

    return flagged_pixels

test_images = np.array([get_toa_texas_patch(path) for path in images_test['tiff_file']])
test_masks = np.array([get_mask_patch(path) for path in masks_test['mask_file']])

y_pred_flat = test_images.flatten()
y_test_flat = test_masks.flatten()

get_metrics_results(y_pred_flat,y_test_flat)
plot_inferences(test_masks, test_images, OUTPUT_PATH, list_entities_plot=[], method="texas", n_images=50)