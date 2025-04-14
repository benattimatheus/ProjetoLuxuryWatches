import os
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from models.service.data_preprocessor import preprocess_data
from models.logs.logger import logger

def train_tpot_model(file_path: str, target_column: str):
    logger.info(f"Carregando os dados do arquivo: {file_path}")
    try:
        data = pd.read_csv(file_path)
        data = data.sample(frac=0.1, random_state=42)  # Usa apenas 0.1% do dataset para acelerar
        logger.info(f"Dados carregados e amostrados com sucesso: {len(data)} linhas.")
    except Exception as e:
        logger.error(f"Erro ao carregar os dados: {e}")
        raise e

    logger.info("Iniciando o pré-processamento dos dados.")
    try:
        X_preprocessed, y, preprocessor = preprocess_data(data, target_column)
        logger.info("Pré-processamento concluído com sucesso.")
    except Exception as e:
        logger.error(f"Erro durante o pré-processamento: {e}")
        raise e

    if X_preprocessed.shape[0] != len(y):
        logger.error(f"Tamanhos diferentes: X={X_preprocessed.shape[0]}, y={len(y)}")
        raise ValueError("Tamanhos de X e y incompatíveis.")
    else:
        logger.info(f"Número de amostras: {X_preprocessed.shape[0]}")

    # Divisão dos dados com uma pequena fatia de teste (0.1%)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.1, random_state=42)
    logger.info(f"Treino: {len(X_train)}, Teste: {len(X_test)}")
    logger.info(f"Tamanho do treino: {len(X_train)} | Tamanho do teste: {len(X_test)}")
    logger.info(f"Valores únicos em y_test: {y_test.unique()}")
    logger.info("Treinando TPOT Regressor (modo rápido).")
    try:

        tpot = TPOTRegressor(
            generations=5,
            population_size=20,
            random_state=42,
            n_jobs=1
        )
        tpot.fit(X_train, y_train)
        logger.info("Modelo treinado com sucesso.")
    except Exception as e:
        logger.error(f"Erro durante o treinamento do TPOT: {e}")
        raise e

    # Avaliação do modelo
    if hasattr(tpot, 'fitted_pipeline_'):
        try:
            score = score = tpot.fitted_pipeline_.score(X_test, y_test)
            logger.info(f"Score no teste: {score}")
            logger.info(f"Pipeline otimizado: {tpot.fitted_pipeline_}")
        except Exception as e:
            logger.error(f"Erro ao avaliar o modelo: {e}")
            raise e
    else:
        logger.error("Pipeline não foi treinado corretamente.")
        return None, None, None

    return tpot, score, tpot.fitted_pipeline_
