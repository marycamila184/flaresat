import utils.landsat_auth
import requests
import pandas as pd
import requests

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

CATEGORY = "urban_areas"
CSV_ATTR = "urban"
URL_LOGIN = "https://m2m.cr.usgs.gov/api/api/json/stable/login"
URL_SCENE_SEARCH = "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search"
DATASET_LANDSAT = "landsat_ot_c2_l1"

list_df_images = []

def make_request(token, item):
        headers = {"X-Auth-Token": token}
        
        body = {
                "datasetName": DATASET_LANDSAT,
                "sceneFilter": {
                    "acquisitionFilter": {"start": item.date_start, "end": item.date_finish},
                    "spatialFilter": {
                        "filterType": "geojson",
                        "geoJson": {
                            "type": "Point",
                            "coordinates": [item.lng, item.lat],
                        },
                    }
                },
                "cloudCoverFilter": {
                    "max": 10,
                    "min": 0,
                    "includeUnknown": "false"
                }
            }
        
        r = requests.post(URL_SCENE_SEARCH, json=body, headers=headers)

        list_results = r.json()["data"]["results"]

        if len(list_results) > 0:
            print(str(len(list_results)) + " scenes found.")
               
            list_data = []

            for element in list_results:
                    data = {
                        "cloud_sat": element["cloudCover"],
                        "entity_id_sat": element["entityId"],
                        "start_date_sat": element["temporalCoverage"]["startDate"],
                        "end_date_sat": element["temporalCoverage"]["endDate"],
                    }

                    list_data.append(data)

            results = pd.DataFrame(list_data)

            results[CSV_ATTR + "_name"] = item.name
            results[CSV_ATTR + "_latitude"] = item.lat
            results[CSV_ATTR + "_longitude"] = item.lng
            results["available_sat"] = True

            list_df_images.append(results)

        else:
            print("No scene found")


path_csv = "/home/marycamila/flaresat/source/" + CATEGORY + "/points_" + CATEGORY + "_valid.csv"
df = pd.read_csv(path_csv)
df = df.head(150)

if "date_start" not in df.columns:
    df["date_start"] = "01-08-2019"
    df["date_finish"] = "31-08-2019"
               
auth_login_time, token = utils.landsat_auth.return_token()

for index, row in df.iterrows():
    make_request(token, row)

df_images = pd.concat(list_df_images, ignore_index=True)
df_images.to_csv("/home/marycamila/flaresat/source/" + CATEGORY + "/scenes_points_" + CATEGORY + ".csv",index=False)
