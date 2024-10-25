
import tarfile
import utils.landsat_auth as auth
import concurrent.futures
from datetime import datetime
import pandas as pd
import requests
import shutil
import os

WORKERS = 4
CATEGORY = "urban_areas"
URL_BASE = "https://m2m.cr.usgs.gov/api/api/json/stable/"
PRODUCT_ID = "5e81f14f92acf9ef"
PATH_IMAGES="/media/marycamila/Expansion/raw/" + CATEGORY

def unzip_tar_files(tar_file_path, entity_id):
    product_list = ["_B1.TIF","_B2.TIF","_B3.TIF","_B4.TIF","_B5.TIF","_B6.TIF","_B7.TIF","_B9.TIF","_B10.TIF","_B11.TIF","_MTL.txt","_QA_PIXEL.TIF"]
    
    path_entity_image = os.path.join(PATH_IMAGES, entity_id) 
        
    with tarfile.open(tar_file_path, 'r') as tar:
        for member in tar.getmembers():
            if any(product_name in member.name for product_name in product_list):
                tar.extract(member, path=path_entity_image)
                    
    print("Extracted file: " + tar_file_path)    
    os.remove(tar_file_path)


def download_single_image(scene, headers):
    url_download_request = URL_BASE + "download-request"
    body = {
        "downloads": [
            {
                "entityId": scene,
                "productId": PRODUCT_ID
            }
        ]
    }

    print("Request sent: " + scene)

    r = requests.post(url_download_request, json=body, headers=headers)
    product = r.json().get("data", {}).get("availableDownloads", [])

    if product:
        url = product[0]["url"]
        if url != "":
            filename = scene
            response = requests.get(url, stream=True, headers=headers)
            if response.status_code == 200:
                save_path = f"/home/marycamila/flaresat/processing/temp/{filename}.tar"

                # Save the downloaded file
                with open(save_path, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)

                print(f"Downloaded scene: {scene}")
                unzip_tar_files(save_path, scene)
            else:
                print(f"Failed to download the file for scene {scene}. Status code: {response.status_code}")
    else:
        print(f"No product available for scene {scene}.")


def download_image(list_scenes):
    auth_login_time, token = auth.return_token()
    print(auth_login_time, token)

    date = datetime.now()
    print("Initiated all downloads: " + str(date))

    for i in range(0, len(list_scenes), WORKERS):
        diff_auth = datetime.now() - auth_login_time
        if diff_auth.total_seconds() / 60 >= 115:
            auth_login_time, token = auth.return_token()

        headers = {'X-Auth-Token': token}
        scenes_to_download = list_scenes[i:i + WORKERS] 
        # Use concurrent downloading
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download_single_image, scene, headers) for scene in scenes_to_download]
            concurrent.futures.wait(futures)

    date = datetime.now()
    print("Completed all downloads: " + str(date))


def main():
    downloaded_images = os.listdir(PATH_IMAGES)
    df_images = pd.read_csv("/home/marycamila/flaresat/source/" + CATEGORY + "/scenes_points_" + CATEGORY + ".csv")    

    total_unique_images = df_images["entity_id_sat"].nunique()
    print(f"Total images for {CATEGORY}: {total_unique_images}")

    df_downloaded = df_images[~df_images["entity_id_sat"].isin(downloaded_images)]
    list_download_unique = df_downloaded["entity_id_sat"].unique()

    remaining_images_count = len(list_download_unique)
    if remaining_images_count == 0:
        print(f"All images for {CATEGORY} have been downloaded.")
        return

    print(f"{CATEGORY} - Remaining images for download: {remaining_images_count}")

    download_image(list_download_unique)


if __name__ == "__main__":
    main()

