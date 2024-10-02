import utils.landsat_auth
from datetime import datetime
import calendar
import requests
import threading
import pandas as pd
import requests

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

URL_LOGIN = "https://m2m.cr.usgs.gov/api/api/json/stable/login"
URL_SCENE_SEARCH = "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search"
REQUESTS_PER_SECOND = 2
DATASET_LANDSAT = "landsat_ot_c2_l1"
CLOUD_COVER = 10

years = ["2019"]
list_df_images = []

# Function to make requests
def make_request(item, token, year, month_str):
    try:
        if item.queue:
            headers = {"X-Auth-Token": token}

            last_day_month = calendar.monthrange(int(year), int(month_str))[1]

            init_date = year + "-" + month_str + "-01"
            end_date = year + "-" + month_str + "-" + str(last_day_month)

            latitude = item.Latitude
            longitude = item.Longitude

            body = {
                "datasetName": DATASET_LANDSAT,
                "sceneFilter": {
                    "acquisitionFilter": {"start": init_date, "end": end_date},
                    "spatialFilter": {
                        "filterType": "geojson",
                        "geoJson": {
                            "type": "Point",
                            "coordinates": [longitude, latitude],
                        },
                    },
                    "cloudCoverFilter": {
                        "max": 10,
                        "min": 0,
                        "includeUnknown": "true"
                    }
                },
            }

            r = requests.post(URL_SCENE_SEARCH, json=body, headers=headers)

            list_results = r.json()["data"]["results"]

            # Remove from queue if the API response is 200
            df.at[item.id, "queue"] = False

            if len(list_results) > 0:
                print(str(len(list_results)) + " scenes found for year: " + year +" - month: " + month_str + " - id: " + str(item.id))
               
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

                results["point_id"] = item.id
                results["point_latitude"] = latitude
                results["point_longitude"] = longitude
                results["point_temp"] = item.Temperature
                results["point_rh"] = item.RadiativeHeat
                results["point_freq"] = item.Frequency
                results["point_area"] = item.Area
                results["point_bcm"] = item.BCM
                results["point_type"] = item.Type                
                results["available_sat"] = True

                list_df_images.append(results)

            else:
                print("No scene found for year: " + year +" - month: " + month_str + " - id: " + str(item.id))
    except Exception as e:
        print("Error for " + str(item.id))
        df.at[item.id, "queue"] = True
        print(e)


for year in years:
    path_csv = "/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv"

    months = [12]
    for month in months:
            month_str = f"{month:02d}"

            try:
                df = pd.read_csv(path_csv, low_memory=False)
                
                # Reset index and rename the index column
                df.reset_index(inplace=True)
                df.rename(columns={"index": "id"}, inplace=True)

                # List to store threads
                threads = []
                 # Get initial tokens and their login times
                auth_login_time1, token1 = utils.landsat_auth.return_token()
                auth_login_time2, token2 = utils.landsat_auth.return_token()

                # Create threads for each request
                for index, row in df.iterrows():
                    diff_auth1 = datetime.now() - auth_login_time1
                    diff_auth2 = datetime.now() - auth_login_time2
                      
                    if diff_auth1.total_seconds() / 60 >= 115:
                        print("Changed token1")
                        auth_login_time1, token1 = utils.landsat_auth.return_token()

                    if diff_auth2.total_seconds() / 60 >= 115:
                        print("Changed token2")
                        auth_login_time2, token2 = utils.landsat_auth.return_token()
                    
                    if index % 2 == 0:
                        token = token1
                    else:
                        token = token2

                    thread = threading.Thread(target=make_request, args=(row, token, year, month_str))
                    threads.append(thread)
                    thread.start()

                    # Limit the number of concurrent threads
                    if len(threads) >= REQUESTS_PER_SECOND:
                        for thread in threads:
                            thread.join()
                        threads = []

                    if len(list_df_images) > 0:
                        df_images = pd.concat(list_df_images, ignore_index=True)
                        df_images.to_csv("/home/marycamila/flaresat/source/landsat_scenes/"+ str(year)+ "/scenes_"+ month_str+ "_.csv",index=False)

            except Exception as e:
                print(e)
                
