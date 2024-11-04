import pandas as pd
import numpy as np
import os

PATH_CSV_POINTS = '/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv'

df_flare = pd.read_csv("/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv")

def check_valid_urban_point(latitude_cat, longitude_cat):
    lat_km = 111.0    
    distance_km = 5.0 
    
    for __, row_flare in df_flare.iterrows():
        lat_flare = row_flare["latitude"]
        lon_flare = row_flare["longitude"]

        lon_diff = distance_km / (lat_km * np.cos(np.radians(lat_flare)))

        if (latitude_cat - distance_km <= lat_flare <= latitude_cat + distance_km) and \
            (longitude_cat - lon_diff <= lon_flare <= longitude_cat + lon_diff):
            return False  # Invalid urban point due to proximity to a flare

    return True  # Valid urban point

df_urban = pd.read_csv("/home/marycamila/flaresat/source/urban_areas/points_urban_areas_full.csv")
df_urban["valid_urban"] = df_urban.apply(lambda x: check_valid_urban_point(x["lat"], x["lng"]), axis=1)

df_urban = df_urban[df_urban["valid_urban"]]
df_urban.drop(columns="valid_urban", inplace=True)

df_urban = df_urban.sort_values("population", ascending=False)
print(df_urban.info())

df_urban.to_csv("/home/marycamila/flaresat/source/urban_areas/points_urban_areas_valid.csv")


