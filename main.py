import os
from controllers.auth_kaggle import AuthKaggle
from controllers.kaggle_repo import KaggleRepository
from models.service.eda_report import EDAReport
from models.logs.logger import logger
from views.services import run_pipeline
from models.service.variable_selection import train_tpot_model  # Importe a função de treinamento do modelo

# Caminhos
credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")
download_path = os.path.expanduser("~/Desktop/ProjetoLuxuryWatches/models/dataset")
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

if __name__ == '__main__':
    file_path = dataset_path  # Caminho para o arquivo CSV
    target = 'price'          # Nome da coluna alvo

    # Chama a função run_pipeline para treinar o modelo e gerar o resumo SHAP
    model, shap_summary = run_pipeline(file_path, target)
    print("Model training and feature analysis completed.")

    # Exibe o gráfico de resumo SHAP
    import shap
    shap.summary_plot(shap_summary.values, shap_summary.data)

    # Chama a função train_tpot_model para treinar o modelo TPOT e obter o pipeline otimizado
    model, score, pipeline = train_tpot_model(file_path, target)
    print(f"Modelo treinado com sucesso. Score: {score}")
    print(f"Pipeline otimizado: {pipeline}")

    # Finaliza o processo
    logger.info("Processo concluído com sucesso.")
