from kaggle.api.kaggle_api_extended import KaggleApi
import os
import logging

logger = logging.getLogger("EDA_Project")

class KaggleRepository:
    def __init__(self, api: KaggleApi):
        self.api = api

    def download_dataset(self, dataset_name: str, path: str):
        try:
            os.makedirs(path, exist_ok=True)
            self.api.dataset_download_files(dataset_name, path=path, unzip=True)
            logger.info(f"Dataset {dataset_name} baixado com sucesso em {path}")
        except Exception as e:
            logger.error(f"Erro ao baixar dataset: {e}")