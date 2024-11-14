import pandas as pd
import numpy as np

PATH_CSV_POINTS = '/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv'
PATH_CSV_URBAN = '/home/marycamila/flaresat/source/urban_areas/points_urban_areas_full.csv'

df_flare = pd.read_csv(PATH_CSV_POINTS)
df_urban = pd.read_csv(PATH_CSV_URBAN)

DISTANCE_KM = 10

df_flare_latitudes = df_flare["latitude"].values
df_flare_longitudes = df_flare["longitude"].values

def get_square_bounds(latitude_cat, longitude_cat, distance_km):
    lat_min = latitude_cat - distance_km / 111.0
    lat_max = latitude_cat + distance_km / 111.0
    lon_min = longitude_cat - distance_km / (111.0 * np.cos(np.radians(latitude_cat)))
    lon_max = longitude_cat + distance_km / (111.0 * np.cos(np.radians(latitude_cat)))
    return lat_min, lat_max, lon_min, lon_max

def is_valid_urban_point(latitude_cat, longitude_cat, distance_km):
    lat_min, lat_max, lon_min, lon_max = get_square_bounds(latitude_cat, longitude_cat, distance_km)

    inside_square = (df_flare_latitudes >= lat_min) & (df_flare_latitudes <= lat_max) & \
                    (df_flare_longitudes >= lon_min) & (df_flare_longitudes <= lon_max)

    return not inside_square.any()

df_urban["valid_urban"] = df_urban.apply(lambda x: is_valid_urban_point(x["lat"], x["lng"], DISTANCE_KM), axis=1)

df_urban = df_urban[df_urban["valid_urban"]]
df_urban.drop(columns="valid_urban", inplace=True)

df_urban = df_urban.sort_values("population", ascending=False)

print(df_urban.info())
df_urban.to_csv("/home/marycamila/flaresat/source/urban_areas/points_urban_areas_valid.csv", index=False)
