
import tarfile
import utils.landsat_auth as auth
from datetime import datetime
import pandas as pd
import requests
import shutil
import os

year = "2019"
month = "03"

URL_BASE = "https://m2m.cr.usgs.gov/api/api/json/stable/"
PRODUCT_ID = "5e81f14f92acf9ef"
PATH_IMAGES="/media/marycamila/Expansion/"

def unzip_tar_files(tar_file_path, entity_id):
    product_list = ["_B1.TIF","_B2.TIF","_B3.TIF","_B4.TIF","_B5.TIF","_B6.TIF","_B7.TIF","_B9.TIF","_B10.TIF","_B11.TIF","_MTL.txt","_QA_PIXEL.TIF"]
    
    path_entity_image = "/media/marycamila/Expansion/raw/" + year + "/" + entity_id
        
    with tarfile.open(tar_file_path, 'r') as tar:
        for member in tar.getmembers():
            if any(product_name in member.name for product_name in product_list):
                tar.extract(member, path=path_entity_image)
                    
    print("Extracted file: " + tar_file_path)    
    os.remove(tar_file_path)

def download_image(list_scenes):
    url_download_request = URL_BASE + "download-request"
    auth_login_time, token = auth.return_token()

    print(auth_login_time, token)

    date = datetime.now()
    print("Initiated all downloads: " + str(date))

    for scene in list_scenes:
        diff_auth = datetime.now() - auth_login_time

        if diff_auth.total_seconds()/60 >= 115:  # 1 hora e 45 min
            auth_login_time, token = auth.return_token()

        headers = {'X-Auth-Token': token}

        body = {
            "downloads": [
                {
                    "entityId": scene,
                    "productId": PRODUCT_ID}
            ]
        }

        r = requests.post(url_download_request, json=body, headers=headers)

        product = r.json()["data"]["availableDownloads"]

        if product:
            url = product[0]["url"]
            if url != "":
                filename = scene
                response = requests.get(url, stream=True, headers=headers)
                if response.status_code == 200:

                    save_path = "/home/marycamila/flaresat/processing/temp/" + filename + ".tar"

                    # Create directory
                    with open(save_path, 'wb') as file:
                        shutil.copyfileobj(response.raw, file)

                    print(f"Downloaded scene: {scene}")
                    unzip_tar_files(save_path, scene)
                else:
                    print(f"Failed to download the file. Status code: {response.status_code}")
    
    date = datetime.now()
    print("Completed all downloads: " + str(date))

def main():
    downloaded_images = os.listdir(PATH_IMAGES + "raw/" + year + "/")
    df_images = pd.read_csv("/home/marycamila/flaresat/source/landsat_scenes/" + year + "/scenes/scenes_" + month + ".csv")    

    print("Total images per month: " + str(len(df_images["entity_id_sat"].unique())))

    df_downloaded = df_images[~df_images["entity_id_sat"].isin(downloaded_images)]
    list_download_unique = df_downloaded["entity_id_sat"].unique()

    len_list = len(list_download_unique)
    print(year + " - Month: " + month + " - Remaining images for download: " + str(len_list))

    download_image(list_download_unique)
    unzip_tar_files()

if __name__ == "__main__":
    main()

