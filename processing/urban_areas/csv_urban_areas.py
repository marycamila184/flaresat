import pandas as pd

df = pd.read_csv("/home/marycamila/flaresat/source/csv_points/worldcities.csv")
df = df[["city", "lat", "lng", "country"]]
df.rename(columns={"lng": "lon"}, inplace=True)

df.to_csv("/home/marycamila/flaresat/source/urban_areas/points_urban_areas.csv")