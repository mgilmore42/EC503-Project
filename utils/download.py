import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(api, dataset_name, destination_path):
    os.makedirs(destination_path, exist_ok=True)

    api.dataset_download_files(dataset_name, path=destination_path, unzip=True)

def download_heart(api):

    # Downloads raw data
    download_dataset(api, 'rashikrahmanpritom/heart-attack-analysis-prediction-dataset', 'data/heart')

def download_housing(api):

    # Downloads raw data
    download_dataset(api, 'codingboss/house-prices-intermediate', 'data/housing')

def download_rain(api):

    # Downloads raw data
    download_dataset(api, 'jsphyg/weather-dataset-rattle-package', 'data/rain')

def download_campus(api):

    # Downloads raw data
    download_dataset(api, 'benroshan/factors-affecting-campus-placement', 'data/campus')



def download_all():
    api = KaggleApi()
    api.authenticate()

    download_heart(api)
    download_housing(api)
    download_rain(api)
    download_campus(api)
