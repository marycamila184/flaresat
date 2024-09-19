import numpy as np
import pandas as pd

from utils.metrics import get_metrics_results
from utils.plot_infereces import plot_inferences
from utils.process_scene_toa import *

OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/nhi_output'
PATH_MTL = '/media/marycamila/Expansion/raw/2019'
PATH_METADATA = '/media/marycamila/Expansion/raw/active_fire/metadata'
MAX_PIXEL_VALUE = 65535

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

# NHI flare reference - https://www.mdpi.com/2072-4292/14/24/6319#B29-remotesensing-14-06319
def get_toa_nhi_patch(file_path):
    img = get_toa_patch(file_path)    
    img = img[:, :, [0,4,5,6]]   

    # Reference https://www.mdpi.com/2072-4292/12/19/3232 and https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
    lswir2 = img [:, :, 3]
    lswir1 = img [:, :, 2]
    lnir = img [:, :, 1]
    b1 = img [:, :, 0]

    nhiswir = (lswir2 - lswir1) / (lswir2 + lswir1)
    nhiswnir = (lswir1 - lnir) / (lswir1 + lnir)

    # Extreme Pixel reference = https://www.mdpi.com/2072-4292/12/19/3232
    extreme_pixel = np.where((lswir1 >= 71.3) & (b1 < 70), 1, 0)
    hp = np.where((nhiswir > 0) | (nhiswnir > 0) | (extreme_pixel > 0), 1, 0)

    return hp

test_images = np.array([get_toa_nhi_patch(path) for path in images_test['tiff_file']])
test_masks = np.array([get_mask_patch(path) for path in masks_test['mask_file']])

y_pred_flat = test_images.flatten()
y_test_flat = test_masks.flatten()

get_metrics_results(y_pred_flat,y_test_flat)
plot_inferences(test_masks, test_images, OUTPUT_PATH, method="nhi")