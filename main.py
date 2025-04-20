# main.py
import os
import argparse
import logging
import shap
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# Controllers
from controllers.auth_kaggle import AuthKaggle
from controllers.kaggle_repo import KaggleRepository

# Models
from models.logs.logger import logger
from models.service.eda_report import EDAReport
from models.service.variable_selection import train_tpot_model

# Views
from views.services import run_pipeline  # <- SHAP apenas

# Adapters
from models.service.dtale_adapter import DtaleAdapter
from models.service.pycaret_adapter import PyCaretAdapter

# Application
from application.use_cases import MLUseCases

# Caminhos padrão
credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")
download_path = os.path.expanduser("models/dataset")
dataset_name = "philmorekoung11/luxury-watch-listings"
dataset_path = os.path.join(download_path, "Watches.csv")

def run_download():
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

    if not os.path.exists(dataset_path):
        logger.error(f"Arquivo {dataset_path} não encontrado após o download.")
        exit()

def generate_eda_reports(dataset_path):
    logger.info("Dataset baixado com sucesso! Gerando relatórios de EDA.")
    eda = EDAReport(dataset_path)
    eda.generate_autoviz()
    eda.generate_dtale()
    eda.generate_ydata()
    logger.info("Relatórios de EDA gerados com sucesso.")

def run_shap_analysis(file_path, target_col):
    logger.info("Executando análise com SHAP (pipeline separado de TPOT)...")
    model, shap_summary = run_pipeline(file_path, target_col)

    if shap_summary is not None:
        logger.info("Gerando gráfico SHAP...")
        shap.summary_plot(shap_summary.values, shap_summary.data)
        plt.show()
        logger.info("Gráfico SHAP exibido.")
    else:
        logger.warning("SHAP summary retornou None. Gráfico não foi gerado.")

def run_tpot_training(file_path, target_col):
    logger.info("Executando treinamento com TPOT...")
    model, score, pipeline = train_tpot_model(file_path, target_col)
    logger.info(f"Modelo TPOT treinado com sucesso. Score: {score}")
    logger.info(f"Pipeline otimizado: {pipeline}")

def main():
    parser = argparse.ArgumentParser(description="ProjetoLuxuryWatches CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Comando: edit
    edit_parser = subparsers.add_parser("edit", help="Abrir dados com Dtale")
    edit_parser.add_argument("csv_filename", help="Arquivo CSV em data/ para editar")

    # Comando: train
    train_parser = subparsers.add_parser("train", help="Treinar modelo com PyCaret")
    train_parser.add_argument("csv_filename", help="CSV em data/ para treinar")
    train_parser.add_argument("target_col", help="Coluna alvo")
    train_parser.add_argument("task_type", help="classification, regression, or clustering")

    args = parser.parse_args()

    dtale_adapter = DtaleAdapter()
    training_adapter = PyCaretAdapter()
    ml_use_cases = MLUseCases(
        dtale_adapter=dtale_adapter,
        training_adapter=training_adapter
    )

    if args.command == "edit":
        logger.info(f"Iniciando modo edição com o arquivo {args.csv_filename}")
        ml_use_cases.edit_data(args.csv_filename)

    elif args.command == "train":
        logger.info(f"Iniciando modo treinamento: arquivo={args.csv_filename}, target={args.target_col}, tipo={args.task_type}")
        ml_use_cases.train_model(args.csv_filename, args.target_col, args.task_type)

    else:
        logger.info("Executando pipeline completa (download, EDA, treino TPOT e análise SHAP)...")
        run_download()
        generate_eda_reports(dataset_path)
        target = "price"

        run_tpot_training(dataset_path, target)
        run_shap_analysis(dataset_path, target)

if __name__ == "__main__":
    main()
