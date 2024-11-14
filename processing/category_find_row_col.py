import glob
import math
import os
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pandas as pd

gdal.UseExceptions()

CATEGORY = "urban_areas"
ATTR = "urban"
PATH_CATEGORY = "/media/marycamila/Expansion/raw/" + CATEGORY
PATCH_SIZE = 256

def create_queue_csv():
    # Setting the original dataset to use q queue
    df_points = pd.read_csv("/home/marycamila/flaresat/source/" + CATEGORY + "/scenes_points_" + CATEGORY + ".csv")
    
    # Some entities have only OLI bandas, and they will be removed (22 scenes)
    df_points = df_points[~df_points["entity_id_sat"].str.contains("LO")]
    
    df_points["row"] = 0
    df_points["col"] = 0
    df_points["col_index"] = 0
    df_points["row_index"] = 0
    df_points["point_processed"] = False

    scenes_downloaded = os.listdir(PATH_CATEGORY)
    df_points = df_points[df_points["entity_id_sat"].isin(scenes_downloaded)]

    # Setting a queue to stop the processing in case it is needed
    df_points.to_csv("/home/marycamila/flaresat/source/" + CATEGORY + "/scenes_points_" + CATEGORY + "_queue.csv", index=False)


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


def main():
    df = pd.read_csv("/home/marycamila/flaresat/source/" + CATEGORY + "/scenes_points_" + CATEGORY + "_queue.csv")
    
    list_images = df["entity_id_sat"].unique()

    print(" ------ Start Processing Entities ------ ")

    rows, cols, row_indices, col_indices = [], [], [], []

    for image_sat in list_images:
        print("Initiated - Entity Id:" + image_sat)

        df_entity = df[df["entity_id_sat"] == image_sat]
        path_image = os.path.join(PATH_CATEGORY, image_sat)
        
        tiff_b1 = glob.glob(os.path.join(path_image, '*_B1.TIF'))[0]
        
        for _, entity in df_entity.iterrows():
            attr_lon = ATTR + "_longitude"
            attr_lat = ATTR + "_latitude"
            row, col, _ = get_row_col(entity[attr_lon], entity[attr_lat], tiff_b1)
            row_index, col_index = calculate_patch_index(row, col)

            rows.append(row)
            cols.append(col)
            row_indices.append(row_index)
            col_indices.append(col_index)

    df["row"] = rows
    df["col"] = cols
    df["row_index"] = row_indices
    df["col_index"] = col_indices     

    df.to_csv("/home/marycamila/flaresat/source/" + CATEGORY + "/scenes_points_" + CATEGORY + "_queue.csv", index=False)

if __name__ == "__main__":
    #create_queue_csv()
    main()