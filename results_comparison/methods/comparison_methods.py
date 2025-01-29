import numpy as np
import matplotlib.pyplot as plt

# TAI gas flare detection reference - https://www.sciencedirect.com/science/article/pii/S1569843222002631
def get_toa_tai(mask_cloud, img):
    img = img[:, :, [4,5,6]]   

    # Reference https://www.sciencedirect.com/science/article/pii/S1569843222002631 and https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
    pnir = img [:, :, 0]
    pnear_swir = img [:, :, 1]
    pfar_swir = img [:, :, 2]
    
    # TAI equation
    tai = (pfar_swir-pnear_swir) / pnir

    # First filter
    flaring_pixels = np.where((tai >= 0.15) & (pfar_swir >= 0.15), 1, 0)

    # In case of NO
    # The saturated pixels are inside de flaring pixels
    saturated_pixels = (pfar_swir > 1) & (pnear_swir > pfar_swir)
    unambigous_fire_pixels = np.where(flaring_pixels | saturated_pixels, 1, 0)
    
    # In case of Yes
    second_filter = (pnear_swir > 0.05) & (pnir > 0.01)

    # Merged all filter results
    flagged_pixels = np.where(second_filter & unambigous_fire_pixels, 1, 0)

    # Remove cloud pixels (exclude flagged pixels that overlap with cloud pixels)
    flare_segmented_pixels = np.where(mask_cloud == 0, flagged_pixels, 0)

    return flare_segmented_pixels