import requests
from datetime import datetime

URL_BASE = "https://m2m.cr.usgs.gov/api/api/json/stable/"

def return_token():
    url_auth = URL_BASE + "login"

    credentials = {"username": "marycamilainfo", "password": "Pink97383822"}
    r = requests.post(url_auth, json=credentials)

    return datetime.now(), r.json()['data']