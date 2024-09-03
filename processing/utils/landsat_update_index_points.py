import pandas as pd

year = "2019"
month= "09"

file_points = "/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv"
file_landsat = "/home/marycamila/flaresat/source/landsat_scenes/"+year+"/scenes_"+month+".csv"

df_landsat = pd.read_csv(file_landsat)
index_update = df_landsat.iloc[-1]["point_id"]

df = pd.read_csv(file_points)
df.loc[:index_update, "queue"] = False

df.to_csv(file_points, index=False)