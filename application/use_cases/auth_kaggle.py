import os
import json
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger("EDA_Project")

class AuthKaggle:
    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        self.api = KaggleApi()

    def authenticate(self):
        try:
            with open(self.credentials_path, 'r') as file:
                dados = json.load(file)
            
            os.environ['KAGGLE_USERNAME'] = dados['username']
            os.environ['KAGGLE_KEY'] = dados['key']

            self.api.authenticate()
            logger.info("Autenticação bem-sucedida!")
            return self.api
        except FileNotFoundError:
            logger.error("Arquivo de credenciais não encontrado!")
        except json.JSONDecodeError:
            logger.error("JSON inválido!")
        except Exception as e:
            logger.error(f"Erro na autenticação: {e}")
        return None