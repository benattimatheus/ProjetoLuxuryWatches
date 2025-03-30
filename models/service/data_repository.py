import pandas as pd
from models.logs.logger import logger

def load_data(file_path: str) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV."""
    logger.info(f"Iniciando o carregamento dos dados do arquivo: {file_path}")
    
    data = pd.read_csv(file_path)
    logger.info(f"Dados carregados com sucesso do arquivo: {file_path}")
    
    # Amostrando 10% dos dados
    data = data.sample(frac=0.1, random_state=1179)
    logger.info("Amostra de 10% dos dados selecionada.")
    
    return data
