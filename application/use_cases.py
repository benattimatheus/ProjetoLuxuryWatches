# application/use_cases.py
import os
import pandas as pd
from ports.dtale_port import DtalePort
from ports.training_port import TrainingPort
from models.service.data_preprocessor import preprocess_data
from models.logs.logger import logger

DATA_FOLDER = "models/dataset"

class MLUseCases:
    def __init__(self, dtale_adapter: DtalePort, training_adapter: TrainingPort):
        self.dtale_adapter = dtale_adapter
        self.training_adapter = training_adapter

    def edit_data(self, csv_filename: str) -> str:
        """Abre o D-Tale para edição do dataset e salva as alterações."""
        full_path = os.path.join(DATA_FOLDER, csv_filename)
        logger.info(f"Lendo arquivo para edição: {full_path}")
        df = pd.read_csv(full_path)

        new_df = self.dtale_adapter.open_in_dtale(df)
        edited_path = os.path.join(DATA_FOLDER, "edited_data.csv")
        new_df.to_csv(edited_path, index=False)
        logger.info(f"Dados editados salvos em {edited_path}")
        return edited_path

    def train_model(self, csv_filename: str, target_col: str, task_type: str):
        """Pré-processa os dados e treina o modelo."""
        full_path = os.path.join(DATA_FOLDER, csv_filename)
        logger.info(f"Lendo dados para treinamento: {full_path}")
        raw_df = pd.read_csv(full_path)

        logger.info("Iniciando pré-processamento dos dados.")
        X_preprocessed_df, y, _ = preprocess_data(raw_df, target_col)
        df_preprocessed = pd.concat([X_preprocessed_df, y.reset_index(drop=True)], axis=1)
        
        logger.info("Iniciando treinamento com PyCaret.")
        model = self.training_adapter.train_model(df_preprocessed, target_col, task_type)

        logger.info("Treinamento finalizado.")
        print(f"Treinamento concluído. Modelo: {model}")
        logger.info("Amostras antes:", raw_df.shape)
        print("Amostras antes:", raw_df.shape)
        logger.info("Amostras depois:", X_preprocessed_df.shape)
        print("Amostras depois:", X_preprocessed_df.shape)
        logger.info("Features:", X_preprocessed_df.columns.tolist())
        print("Features:", X_preprocessed_df.columns.tolist())
        logger.info("Alguma feature constante?", X_preprocessed_df.nunique())
        print("Alguma feature constante?", X_preprocessed_df.nunique())