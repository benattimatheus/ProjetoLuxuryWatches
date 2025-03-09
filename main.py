import os
from application.use_cases.auth_kaggle import AuthKaggle
from adapters.kaggle_repo import KaggleRepository
from application.use_cases.eda_report import EDAReport
from infrastructure.logger import logger

# Caminhos
credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")
download_path = os.path.expanduser("~/Desktop/ProjetoLuxuryWatches/content")
dataset_name = "philmorekoung11/luxury-watch-listings"
dataset_path = os.path.join(download_path, "Watches.csv")

# Autenticação
logger.info("Iniciando autenticação no Kaggle.")
auth = AuthKaggle(credentials_path)
api = auth.authenticate()

if api:
    logger.info("Autenticação bem-sucedida! Fazendo download do dataset.")
    repo = KaggleRepository(api)
    repo.download_dataset(dataset_name, download_path)
else:
    logger.error("Falha na autenticação do Kaggle. Encerrando aplicação.")
    exit()

# Verifica se o dataset foi baixado
if not os.path.exists(dataset_path):
    logger.error(f"Arquivo {dataset_path} não encontrado após o download.")
    exit()

# Análise exploratória
logger.info("Dataset baixado com sucesso! Gerando relatórios de EDA.")
eda = EDAReport(dataset_path)
eda.generate_autoviz()
#eda.generate_sweetviz()
eda.generate_dtale()
eda.generate_ydata()

logger.info("Processo concluído com sucesso.")