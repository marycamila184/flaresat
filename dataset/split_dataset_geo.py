from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 

PATH_DATASET = '/home/marycamila/flaresat/dataset'

PATH_FLARE_PATCHES = '/home/marycamila/flaresat/dataset/flare_patches'
PATH_FLARE_MASKS = '/home/marycamila/flaresat/dataset/flare_mask_patches'

PATH_FIRE_PATCHES = '/home/marycamila/flaresat/dataset/fire_patches'
PATH_FIRE_MASKS = '/home/marycamila/flaresat/dataset/fire_mask_patches'

PATH_VOLCANOES_PATCHES = '/home/marycamila/flaresat/dataset/volcanoes_patches'
PATH_VOLCANOES_MASKS = '/home/marycamila/flaresat/dataset/volcanoes_mask_patches'

PATH_URBAN_PATCHES = '/home/marycamila/flaresat/dataset/urban_areas_patches'
PATH_URBAN_MASKS = '/home/marycamila/flaresat/dataset/urban_areas_mask_patches'

PATH_CSV_ACTIVE_FIRE = '/home/marycamila/flaresat/source/landsat_scenes/2019/active_fire'
PATH_CSV_POINTS = '/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv'

TRAIN_RATIO = 0.70
TEST_RATIO = 0.30
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def get_countries():
    list_valid_patches = os.listdir(PATH_FLARE_PATCHES)
    entity_row_col = []
    for valid_patch in list_valid_patches:
        patch = valid_patch.split('_')[1:4]
        entity = patch[0]
        row, col = int(patch[1]), int(patch[2])
        entity_row_col.append([entity, row, col])
    
    # Getting entity id and points
    list_csv_active_fire = os.listdir(PATH_CSV_ACTIVE_FIRE)
    list_fire_patches = []
    for file in list_csv_active_fire:
        file_name = os.path.join(PATH_CSV_ACTIVE_FIRE, file)
        df_scenes = pd.read_csv(file_name)
        
        mask = df_scenes.apply(lambda row: [row['entity_id_sat'], row['row_index'], row['col_index']] in entity_row_col, axis=1)
        df_scenes = df_scenes[mask]

        unique_pairs = df_scenes[['entity_id_sat', 'row_index', 'col_index', 'point_id_number']].drop_duplicates()
        list_fire_patches.append(unique_pairs)

    df_flares = pd.concat(list_fire_patches, ignore_index=True)
    
    df_countries = pd.read_csv(PATH_CSV_POINTS)
    df_countries.reset_index(inplace=True)

    df_flare_countries = df_flares.merge(df_countries, how='inner', left_on='point_id_number', right_on=df_countries.index)
    df_flare_countries = df_flare_countries[['cntry_name', 'flr_type', 'latitude','longitude','point_id_number', 'entity_id_sat', 'row_index', 'col_index']]

    return df_flare_countries

list_patches = os.listdir(PATH_FLARE_PATCHES)
df = get_countries()

grouped = df.groupby(['cntry_name', 'latitude','longitude', 'row_index', 'col_index']).size().reset_index(name='count')
len_test = int(len(list_patches) * TEST_RATIO)

# Test dataset will have patches which are unique ('latitude','longitude', 'row_index', 'col_index') frequency == 1
test_group = grouped[grouped["count"] == 1]
test_group = test_group.sample(n=len_test)

mask = df.set_index(['cntry_name', 'latitude', 'longitude', 'row_index', 'col_index']).index.isin(
    test_group.set_index(['cntry_name', 'latitude', 'longitude', 'row_index', 'col_index']).index
)

# Unique patches for test dataset
df_test = df[mask]
df_test = df_test[["entity_id_sat", "row_index", "col_index"]].drop_duplicates()

# Load patch data
list_patches = os.listdir(PATH_FLARE_PATCHES)
data = [filename.split("_")[1:4] for filename in list_patches]
df_patches = pd.DataFrame(data, columns=["entity_id_sat", "row_index", "col_index"])

df_test[["entity_id_sat", "row_index", "col_index"]] = df_test[["entity_id_sat", "row_index", "col_index"]].astype(str)
df_patches[["entity_id_sat", "row_index", "col_index"]] = df_patches[["entity_id_sat", "row_index", "col_index"]].astype(str)

df_test_patches = pd.merge(df_test, df_patches, on=["entity_id_sat", "row_index", "col_index"], how="inner")
df_test_patches = df_test_patches[["entity_id_sat", "row_index", "col_index"]].copy()

# ---- TEST PATCHES

# Test list image for future comparison.
# list_test_images = df_test_patches["entity_id_sat"].unique()
# df_test_images = pd.DataFrame(list_test_images, columns=["entity_id_test"])
# df_test_images.to_csv(os.path.join(PATH_DATASET, 'entity_id_test.csv'), index=False)

df_test_patches["tiff_file"] = "fire_" + df_test_patches["entity_id_sat"] + "_" + df_test_patches["row_index"] + "_" + df_test_patches["col_index"] + "_patch.tiff"
df_test_patches["mask_file"] = "fire_" + df_test_patches["entity_id_sat"] + "_" + df_test_patches["row_index"] + "_" + df_test_patches["col_index"] + "_mask.tiff"

df_test_patches["tiff_file"] = df_test_patches["tiff_file"].apply(lambda x: os.path.join(PATH_FLARE_PATCHES, x))
df_test_patches["mask_file"] = df_test_patches["mask_file"].apply(lambda x: os.path.join(PATH_FLARE_MASKS, x))

x_test = df_test_patches[['tiff_file']]
y_test = df_test_patches[['mask_file']]

# ---- TRAIN FLARE PATCHES

df_train_patches= df_patches[~df_patches.set_index(["entity_id_sat", "row_index", "col_index"]).index.isin(df_test_patches.set_index(["entity_id_sat", "row_index", "col_index"]).index)].copy()

df_train_patches["tiff_file"] = "fire_" + df_train_patches["entity_id_sat"] + "_" + df_train_patches["row_index"] + "_" + df_train_patches["col_index"] + "_patch.tiff"
df_train_patches["mask_file"] = "fire_" + df_train_patches["entity_id_sat"] + "_" + df_train_patches["row_index"] + "_" + df_train_patches["col_index"] + "_mask.tiff"

df_train_patches["tiff_file"] = df_train_patches["tiff_file"].apply(lambda x: os.path.join(PATH_FLARE_PATCHES, x))
df_train_patches["mask_file"] = df_train_patches["mask_file"].apply(lambda x: os.path.join(PATH_FLARE_MASKS, x))

df_train_patches, df_val_patches = train_test_split(df_train_patches, test_size=0.15, random_state=RANDOM_STATE)

x_train = df_train_patches[['tiff_file']]
y_train = df_train_patches[['mask_file']]
x_val = df_val_patches[['tiff_file']]
y_val = df_val_patches[['mask_file']]

# ---- FIRE DATASET

list_active_fire_patches = os.listdir(PATH_FIRE_PATCHES)
list_fire = []
for patche_fire in list_active_fire_patches:
    path = os.path.join(PATH_FIRE_PATCHES, patche_fire)
    mask = os.path.join(PATH_FIRE_MASKS, patche_fire)
    new_row = {"tiff_file": path, "mask_file": mask}
    list_fire.append(new_row)

df_fire = pd.DataFrame(list_fire)
x_train_fire, x_temp_fire, y_train_fire, y_temp_fire = train_test_split(df_fire['tiff_file'], df_fire['mask_file'], test_size=0.4)
x_val_fire, x_test_fire, y_val_fire, y_test_fire = train_test_split(x_temp_fire, y_temp_fire, test_size=0.5)

# ---- VOLCANOES DATASET

list_volcanoes_patches = os.listdir(PATH_VOLCANOES_PATCHES)
list_volcanoes = []

for patch_volcano in list_volcanoes_patches:
    path = os.path.join(PATH_VOLCANOES_PATCHES, patch_volcano)
    mask_filename = patch_volcano.replace("patch.tiff", "mask.tiff")
    mask = os.path.join(PATH_VOLCANOES_MASKS, mask_filename)
    new_row = {"tiff_file": path, "mask_file": mask}
    list_volcanoes.append(new_row)

df_volcanoes = pd.DataFrame(list_volcanoes)
x_test_volcano = df_volcanoes[['tiff_file']]
y_test_volcano = df_volcanoes[['mask_file']]

# ---- URBAN AREAS DATASET

# ler o df das cenas com os pontos 
# agrupar por cidade
# pegar a lista e fazer um split pensando no numero final de imagens
# split de treino, validacao e teste

list_urban_patches = os.listdir(PATH_URBAN_PATCHES)
list_urban = []

for patch_urban in list_urban_patches:
    path = os.path.join(PATH_URBAN_PATCHES, patch_urban)
    mask_filename = patch_urban.replace("patch.tiff", "mask.tiff")
    mask = os.path.join(PATH_URBAN_MASKS, mask_filename)
    new_row = {"tiff_file": path, "mask_file": mask}
    list_urban.append(new_row)

df_urban_areas = pd.DataFrame(list_urban)
x_test_urban = df_urban_areas[['tiff_file']]
y_test_urban = df_urban_areas[['mask_file']]

# ---- MERGE FLARE, FIRE AND VOLCANOES DATASETS

x_train = pd.concat([x_train, x_train_fire])
y_train = pd.concat([y_train, y_train_fire])

x_val = pd.concat([x_val, x_val_fire])
y_val = pd.concat([y_val, y_val_fire])

x_test = pd.concat([x_test, x_test_fire, x_test_volcano, x_test_urban])
y_test = pd.concat([y_test, y_test_fire, y_test_volcano, y_test_urban])

x_train.to_csv(os.path.join(PATH_DATASET, 'images_train.csv'), index=False)
y_train.to_csv(os.path.join(PATH_DATASET, 'masks_train.csv'), index=False)
x_val.to_csv(os.path.join(PATH_DATASET, 'images_val.csv'), index=False)
y_val.to_csv(os.path.join(PATH_DATASET, 'masks_val.csv'), index=False)
x_test.to_csv(os.path.join(PATH_DATASET, 'images_test.csv'), index=False)
y_test.to_csv(os.path.join(PATH_DATASET, 'masks_test.csv'), index=False)

# Adding volcanoes to the train and val

# Test performed to see how is the performance when added volcanoes to the dataset
#x_train_volcano, x_temp_volcano, y_train_volcano, y_temp_volcano = train_test_split(df_volcanoes['tiff_file'], df_volcanoes['mask_file'], test_size=0.4)
#x_val_volcano, x_test_volcano, y_val_volcano, y_test_volcano = train_test_split(x_temp_volcano, y_temp_volcano, test_size=0.5)

# Test - only fire patches
# x_test_fire.to_csv(os.path.join(PATH_DATASET, 'images_fire_test.csv'), index=False)
# y_test_fire.to_csv(os.path.join(PATH_DATASET, 'images_fire_mask.csv'), index=False)

# # Teste - only volcanoes patches
# x_test_volcano.to_csv(os.path.join(PATH_DATASET, 'images_volcanoes_test.csv'), index=False)
# y_test_volcano.to_csv(os.path.join(PATH_DATASET, 'images_volcanoes_mask.csv'), index=False)

# # Teste - only volcanoes patches
# x_test_volcano.to_csv(os.path.join(PATH_DATASET, 'images_volcanoes_test.csv'), index=False)
# y_test_volcano.to_csv(os.path.join(PATH_DATASET, 'images_volcanoes_mask.csv'), index=False)