# In application/services.py
from models.service.data_repository import load_data
from models.service.data_preprocessor import preprocess_data
from models.service.model_service import build_model
from models.service.feature_service import analyze_features
import pandas as pd
from models.logs.logger import logger

def run_pipeline(file_path: str, target: str):
    logger.info(f"Iniciando o pipeline com o arquivo: {file_path} e a variável alvo: {target}")

    # Load raw data
    logger.info("Carregando os dados brutos.")
    data = load_data(file_path)
    logger.info(f"Dados carregados com sucesso. Shape dos dados: {data.shape}")
    
    # Preprocess data
    logger.info("Iniciando o pré-processamento dos dados.")
    X_preprocessed, y, preprocessor = preprocess_data(data, target)
    logger.info("Pré-processamento dos dados concluído.")
    
    # Create a new DataFrame with preprocessed features
    X_df = pd.DataFrame(X_preprocessed)
    data_preprocessed = pd.concat([X_df, y.reset_index(drop=True)], axis=1)
    logger.info(f"Novo DataFrame de dados pré-processados criado com sucesso. Shape: {data_preprocessed.shape}")
    
    # Build and train the model using the preprocessed data.
    logger.info("Iniciando a construção e treinamento do modelo.")
    model, le = build_model(data_preprocessed, target)
    if model is not None:
        logger.info(f"Modelo treinado com sucesso: {model}")
    else:
        logger.warning("Falha no treinamento do modelo.")
    
    # Analyze features with the trained model.
    logger.info("Iniciando a análise das features com o modelo treinado.")
    shap_summary = analyze_features(model, data_preprocessed, target)
    logger.info("Análise das features concluída.")

    return model, shap_summary
