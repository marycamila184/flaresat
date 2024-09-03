import tifffile as tiff
from shapely.geometry import box
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np

YEAR = "2019"
MONTH = "03"

PATH_FIRE = "/home/marycamila/flaresat/fire_mask/fire_images/" + YEAR + "/" + MONTH
PATH_CSV = "/home/marycamila/flaresat/source/landsat_scenes"
HALF_SIZE_SQUARE = 35

def draw_square_mask(df_points_fire, entity):
    df_points_fire = df_points_fire[df_points_fire["entity_id_sat"] == entity]

    list_rows = df_points_fire.row_index.unique()

    for row_index in list_rows:
        df_grouped = df_points_fire[df_points_fire["row_index"] == row_index]
        list_cols = df_grouped.col_index.unique()

        for col_index in list_cols:
            prefix = "fire_" + entity + "_" + str(row_index) + "_" + str(col_index)
            path = f"{PATH_FIRE}/{prefix}_mask.tiff"

            image = tiff.imread(path)

            image = np.resize(image, (256, 256, 1))

            image_pil = Image.fromarray(image[:, :, 0])
            draw = ImageDraw.Draw(image_pil)

            df_points_flter = df_points_fire[(df_points_fire["row_index"] == row_index) & (df_points_fire["col_index"] == col_index)]

            squares = []
            for j, item in df_points_flter.iterrows():
                center_x = item.col - (256 * item.col_index)
                center_y = item.row - (256 * item.row_index)

                window_size_add = int((item["point_area"] * 2) / 30)

                square = box(center_x - (HALF_SIZE_SQUARE + window_size_add), 
                             center_y - (HALF_SIZE_SQUARE + window_size_add), 
                             center_x + (HALF_SIZE_SQUARE + window_size_add), 
                             center_y + (HALF_SIZE_SQUARE + window_size_add))

                minx, miny, maxx, maxy = square.bounds
                draw.rectangle([minx, miny, maxx, maxy], outline=255, width=1)

            prefix += "_square.tif"
            image_np = np.array(image_pil)
            tiff.imwrite(f"{PATH_FIRE}/{prefix}", image_np)

if __name__ == "__main__":
    df = pd.read_csv(f"{PATH_CSV}/{YEAR}/active_fire/scenes_{MONTH}_queue.csv")
    fire_points_df = df[df["pixels_count_fire"] > 0]

    list_scenes = fire_points_df["entity_id_sat"].unique()

    for entity in list_scenes:
        draw_square_mask(fire_points_df, entity)