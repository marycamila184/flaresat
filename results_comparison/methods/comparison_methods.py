import numpy as np
import matplotlib.pyplot as plt

# NHI flare reference - https://ieeexplore.ieee.org/document/9681815
# def get_toa_nhi(img):
#     img = img[:, :, [4,5,6]]   

#     # Reference https://ieeexplore.ieee.org/document/9681815 and https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
#     lswir2 = img [:, :, 2]
#     lswir1 = img [:, :, 1]
#     lnir = img [:, :, 0]

#     nhiswir = (lswir2 - lswir1) / (lswir2 + lswir1)
#     nhiswnir = (lswir1 - lnir) / (lswir1 + lnir)

#     hp = np.where((nhiswir > 0) | (nhiswnir > 0), 1, 0)

#     return hp


# https://doi.org/10.3390/su15065333 
# def calculate_global_rxd(scene, channels, entity):
#     X = scene.reshape(-1, channels)
    
#     mu = np.mean(X, axis=0)
    
#     cov_matrix = np.cov(X, rowvar=False)
#     cov_matrix_inv = np.linalg.inv(cov_matrix)

#     diffs = X - mu

#     # Reference https://www.mdpi.com/2071-1050/15/6/5333 - 3. The Reed–Xiaoli Detector Method
#     distances = np.sqrt(np.einsum('ij,ij->i', diffs @ cov_matrix_inv, diffs))

#     distances_image = distances.reshape(scene.shape[0], scene.shape[1])

#     rxd_scene_img = ((distances_image - distances_image.min()) / 
#                      (distances_image.max() - distances_image.min()) * 255).astype(np.uint8)
    
#     cv2.imwrite(os.path.join(OUTPUT_PATH, 'test_plot', f'scene_rxd_{entity}.png'), rxd_scene_img)

#     mask_scene = distances_image > THRESHOLD

#     return mask_scene

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