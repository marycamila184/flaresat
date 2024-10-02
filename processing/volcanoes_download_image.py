
import tarfile
import utils.landsat_auth as auth
from datetime import datetime
import pandas as pd
import requests
import shutil
import os

URL_BASE = "https://m2m.cr.usgs.gov/api/api/json/stable/"
PRODUCT_ID = "5e81f14f92acf9ef"
PATH_IMAGES="/media/marycamila/Expansion/raw/volcanoes"

def unzip_tar_files(tar_file_path, entity_id):
    product_list = ["_B1.TIF","_B2.TIF","_B3.TIF","_B4.TIF","_B5.TIF","_B6.TIF","_B7.TIF","_B9.TIF","_B10.TIF","_B11.TIF","_MTL.txt","_QA_PIXEL.TIF"]
    
    path_entity_image = os.path.join(PATH_IMAGES, entity_id) 
        
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
    list_entities_downloaded = os.listdir(PATH_IMAGES)
    df_images = pd.read_csv("/home/marycamila/flaresat/source/volcanoes/scenes_points_volcanoes.csv") 
    list_download_unique = df_images["entity_id_sat"].unique()

    list_scenes_to_download = list(set(list_download_unique) - set(list_entities_downloaded))
    download_image(list_scenes_to_download)

if __name__ == "__main__":
    main()

