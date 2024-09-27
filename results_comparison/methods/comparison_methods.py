import numpy as np

# NHI flare reference - https://ieeexplore.ieee.org/document/9681815
def get_toa_nhi(img):
    img = img[:, :, [4,5,6]]   

    # Reference https://ieeexplore.ieee.org/document/9681815 and https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
    lswir2 = img [:, :, 2]
    lswir1 = img [:, :, 1]
    lnir = img [:, :, 0]

    nhiswir = (lswir2 - lswir1) / (lswir2 + lswir1)
    nhiswnir = (lswir1 - lnir) / (lswir1 + lnir)

    hp = np.where((nhiswir > 0) | (nhiswnir > 0), 1, 0)

    return hp

# Texas gas flare detection reference - https://www.sciencedirect.com/science/article/pii/S1569843222002631
def get_toa_texas(img):
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