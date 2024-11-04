import glob
import math
import os
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pandas as pd

gdal.UseExceptions()

PATH_RAW = "/media/marycamila/Expansion/raw"
PATH_CSV = "/home/marycamila/flaresat/source/landsat_scenes"
PATCH_SIZE = 256

YEAR = "2019"
MONTH = "12"

def get_row_col(long, lat, path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    target = osr.SpatialReference(wkt=ds.GetProjection())

    width = ds.RasterXSize

    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lat, long)

    point.Transform(transform)

    geotransform = ds.GetGeoTransform()

    # Calculate the column (X) and row (Y) indices
    col = int((point.GetX() - geotransform[0]) / geotransform[1])
    row = int((geotransform[3] - point.GetY()) / abs(geotransform[5]))

    return row, col, width


def calculate_patch_index(row, col):
    col_index = math.floor(col // PATCH_SIZE)
    row_index = math.floor(row // PATCH_SIZE)

    return row_index, col_index


def create_queue_csv():
    # Setting the original dataset to use q queue
    df_points = pd.read_csv(PATH_CSV + "/" + YEAR + "/scenes/scenes_" + MONTH + ".csv")
    
    # Some entities have only OLI bandas, and they will be removed (22 scenes)
    df_points = df_points[~df_points["entity_id_sat"].str.contains("LO")]
    
    df_points["row"] = 0
    df_points["col"] = 0
    df_points["col_index"] = 0
    df_points["row_index"] = 0
    df_points["point_processed"] = False
    # Setting a queue to stop the processing in case it is needed
    df_points.to_csv(PATH_CSV + "/" + YEAR + "/active_fire/scenes_" + MONTH + "_queue.csv", index=False)


def main():
    # Getting already processed points
    df = pd.read_csv(PATH_CSV + "/" + YEAR + "/active_fire/scenes_" + MONTH + "_queue.csv")
    
    # Pegando somente os pontos validos.
    df_points = df[~df["point_processed"]]

    list_images = df_points["entity_id_sat"].unique()

    print(" ------ Start Processing Entities ------ ")

    for image_sat in list_images:
        print("Initiated - Entity Id:" + image_sat)

        df_entity = df_points[df_points["entity_id_sat"] == image_sat]
        path_image = PATH_RAW + "/" + YEAR + "/" + image_sat
        tiff_b1 = glob.glob(os.path.join(path_image, '*_B1.TIF'))[0]

        for entity in df_entity.itertuples():
            row, col, _ = get_row_col(entity.point_longitude, entity.point_latitude, tiff_b1)
            row_index, col_index = calculate_patch_index(row, col)
            df.loc[entity.Index, "row"] = row
            df.loc[entity.Index, "col"] = col
            df.loc[entity.Index, "row_index"] = row_index
            df.loc[entity.Index, "col_index"] = col_index
            df.at[entity.Index,'point_processed'] = True        

        df.to_csv(PATH_CSV + "/" + YEAR + "/active_fire/scenes_" + MONTH + "_queue.csv", index=False)
        print("Finished - Entity Id:" + image_sat)

if __name__ == "__main__":
    #create_queue_csv()
    main()