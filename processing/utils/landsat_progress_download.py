import pandas as pd
import shutil
import os

PATH_HD_IMAGES="/media/marycamila/Expansion/raw/2019"

df03 = pd.read_csv("/home/marycamila/flaresat/source/landsat_scenes/2019/scenes/scenes_03.csv")
df08 = pd.read_csv("/home/marycamila/flaresat/source/landsat_scenes/2019/scenes/scenes_08.csv")
df09 = pd.read_csv("/home/marycamila/flaresat/source/landsat_scenes/2019/scenes/scenes_09.csv")
df12 = pd.read_csv("/home/marycamila/flaresat/source/landsat_scenes/2019/scenes/scenes_12.csv")

df = pd.concat([df03, df08, df09, df12])

list_entities = df["entity_id_sat"].unique()
list_dir = os.listdir(PATH_HD_IMAGES)

print(len(list_dir))

print("Imagens já baixadas:")
print(len(df[df["entity_id_sat"].isin(list_dir)]["entity_id_sat"].unique()))

# Print the number of images still to be downloaded
print("Imagens para baixar:")
print(len(df[~df["entity_id_sat"].isin(list_dir)]["entity_id_sat"].unique()))

# def remove_old_entities():
#     list_dir = os.listdir(PATH_HD_IMAGES)  # List all files in the directory
#     list_dir_series = pd.Series(list_dir)  # Convert to a pandas Series
    
#     # Check which items in list_dir are not in list_entities
#     old_entities = list_dir_series[~list_dir_series.isin(list_entities)]
    
#     print(f"Imagens a serem deletadas: {len(old_entities)}")
    
#     # Iterate over the old entities and remove them
#     for entity in old_entities:
#         dir_path = os.path.join(PATH_HD_IMAGES, entity)
#         try:
#             shutil.rmtree(dir_path)
#             print(f"Deleted: {entity}")
#         except Exception as e:
#             print(f"Error deleting {entity}: {e}")

# remove_old_entities()
