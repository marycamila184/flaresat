import utils.landsat_auth
from datetime import datetime
import calendar
import requests
import threading
import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

URL_LOGIN = "https://m2m.cr.usgs.gov/api/api/json/stable/login"
URL_SCENE_SEARCH = "https://m2m.cr.usgs.gov/api/api/json/stable/scene-search"
REQUESTS_PER_SECOND = 2
DATASET_LANDSAT = "landsat_ot_c2_l1"
CLOUD_COVER = 10

years = ["2019"]
months = ["12"]
list_df_images = []
path_csv = "/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv"
df = pd.read_csv(path_csv, low_memory=False)


def make_request(item, token, year, month_str, timeout=10):
    try:
        headers = {"X-Auth-Token": token}

        # Get the last day of the month
        last_day_month = calendar.monthrange(int(year), int(month_str))[1]
        init_date = f"{year}-{month_str}-01"
        end_date = f"{year}-{month_str}-{last_day_month:02d}"

        # Create request body
        body = {
            "datasetName": DATASET_LANDSAT,
            "sceneFilter": {
                "acquisitionFilter": {"start": init_date, "end": end_date},
                "spatialFilter": {
                    "filterType": "geojson",
                    "geoJson": {
                        "type": "Point",
                        "coordinates": [item.longitude, item.latitude],
                    },
                },
                "cloudCoverFilter": {
                    "max": 15,
                    "min": 0,
                    "includeUnknown": False
                }
            },
        }

        # Make POST request with timeout
        r = requests.post(URL_SCENE_SEARCH, json=body, headers=headers, timeout=timeout)

        df.loc[df["id_number"] == item.id_number, "queue"] = False

        # Check if the response is successful
        if r.status_code == 200:
            data = r.json()

            # Ensure "data" and "results" exist
            if "data" in data and "results" in data["data"]:
                list_results = data["data"]["results"]

                if len(list_results) > 0:
                    logging.info(f"{len(list_results)} scenes found for year: {year} - month: {month_str} - id: {item.id_number}")

                    list_data = []
                    for element in list_results:
                        # Check if necessary fields are present in the response
                        cloud_cover = element.get("cloudCover", None)
                        entity_id = element.get("entityId", None)
                        temporal_coverage = element.get("temporalCoverage", {})

                        start_date = temporal_coverage.get("startDate", None)
                        end_date = temporal_coverage.get("endDate", None)

                        data = {
                            "cloud_sat": cloud_cover,
                            "entity_id_sat": entity_id,
                            "start_date_sat": start_date,
                            "end_date_sat": end_date,
                        }

                        list_data.append(data)

                    results = pd.DataFrame(list_data)

                    # Add additional item information
                    results["point_id_number"] = item.id_number
                    results["point_latitude"] = item.latitude
                    results["point_longitude"] = item.longitude
                    results["point_temp"] = item.avg_temp
                    results["point_freq"] = item.dtc_freq
                    results["point_ellip"] = item.ellip
                    results["point_flr_volume"] = item.flr_volume
                    results["point_type"] = item.flr_type                
                    results["available_sat"] = True

                    list_df_images.append(results)

                else:
                    logging.info(f"No scene found for year: {year} - month: {month_str} - id: {item.id_number}")
            else:
                logging.error("Unexpected response format: 'data' or 'results' not found")
        else:
            df.loc[df["id_number"] == item.id_number, "queue"] = True
            df.to_csv(path_csv, index=False)
            logging.error(f"Failed request with status code {r.status_code} for id {item.id_number}")

    except requests.Timeout:
        df.loc[df["id_number"] == item.id_number, "queue"] = True
        logging.error(f"Request timeout for id {item.id_number}")
        df.to_csv(path_csv, index=False)
    except requests.RequestException as e:
        df.loc[df["id_number"] == item.id_number, "queue"] = True
        logging.error(f"Request failed for id {item.id_number}: {e}")
        df.to_csv(path_csv, index=False)
    except Exception as e:
        df.loc[df["id_number"] == item.id_number, "queue"] = True
        logging.error(f"Error processing request for id {item.id_number}: {e}")
        df.to_csv(path_csv, index=False)    
    

for year in years:
    for month_str in months:
            try:
                threads = []
                auth_login_time1, token1 = utils.landsat_auth.return_token()
                auth_login_time2, token2 = utils.landsat_auth.return_token()

                # Create threads for each request
                df_filtered = df[df["queue"]]

                for index, row in df_filtered.iterrows():
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
                        df_images.to_csv("/home/marycamila/flaresat/source/landsat_scenes/"+ str(year)+ "/scenes/scenes_"+ month_str+ "_.csv",index=False)

            except Exception as e:
                print(e)
                
