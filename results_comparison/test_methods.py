import numpy as np
import pandas as pd

from methods.comparison_methods import *
from utils.plot_infereces import plot_scene_and_squared
from utils.process_scene_toa import *

OUTPUT_PATH = '/home/marycamila/flaresat/results_comparison/output/'
PATH_SOURCE = '/home/marycamila/flaresat/results_comparison/source/'

methods = ['nhi', 'texas']

scenes = pd.read_csv('/home/marycamila/flaresat/results_comparison/source/test_scenes_points.csv', delimiter=';')

for method in methods:
    df_scenes = scenes[scenes["method"] == method]

    for index, scene_row in df_scenes.iterrows():
        scene_path = os.path.join(PATH_SOURCE, method, scene_row['folder']) 
        
        if method == 'nhi':
            # NHI flare reference - https://ieeexplore.ieee.org/document/9681815
            scene = get_toa_scene(scene_path, 'RADIANCE')
            # processed_scene = get_toa_nhi(scene)
        else:
            # Texas gas flare detection reference - https://www.sciencedirect.com/science/article/pii/S1569843222002631
            scene = get_toa_scene(scene_path, 'REFLECTANCE')
            processed_scene = get_toa_texas(scene)

        lon, lat = scene_row['lon'], scene_row['lat']
        
        tiff_b1 = glob.glob(os.path.join(scene_path, '*_B1.TIF'))[0]
        row, col = get_row_col(lon, lat, tiff_b1)
        
        output_path = os.path.join(OUTPUT_PATH, method) 
        plot_scene_and_squared(scene, processed_scene, scene_row['location'], row, col, output_path, method)