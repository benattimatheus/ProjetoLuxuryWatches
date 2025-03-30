import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from models.service.data_preprocessor import preprocess_data  # Importando a função de pré-processamento
from models.logs.logger import logger  # Importando o logger

def train_tpot_model(file_path: str, target_column: str):
    """
    Treina um modelo TPOT usando um conjunto de dados e uma coluna alvo especificada.
    
    Parâmetros:
      - file_path: Caminho para o arquivo CSV de entrada.
      - target_column: Nome da coluna alvo para a previsão.
    
    Retorna:
      - model: O modelo TPOT treinado.
      - score: A pontuação do modelo no conjunto de teste.
      - pipeline: O pipeline otimizado encontrado pelo TPOT.
    """
    # Carregar os dados
    logger.info(f"Carregando os dados do arquivo: {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info("Dados carregados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao carregar os dados: {e}")
        raise e

    # Chama o pré-processamento dos dados
    logger.info("Iniciando o pré-processamento dos dados.")
    try:
        X_preprocessed, y, preprocessor = preprocess_data(data, target_column)
        logger.info("Pré-processamento concluído com sucesso.")
    except Exception as e:
        logger.error(f"Erro durante o pré-processamento: {e}")
        raise e

    # Verificar se X e y têm o mesmo número de amostras
    if X_preprocessed.shape[0] != len(y):
        logger.error(f"X e y têm tamanhos diferentes: X={X_preprocessed.shape[0]}, y={len(y)}")
        raise ValueError(f"X e y têm tamanhos diferentes: X={X_preprocessed.shape[0]}, y={len(y)}")
    else:
        logger.info(f"X e y têm o mesmo número de amostras: {X_preprocessed.shape[0]} amostras.")

    # Dividir os dados em treino e teste
    logger.info("Dividindo os dados em conjuntos de treino e teste.")
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)
    logger.info(f"Dados divididos: {len(X_train)} amostras de treino, {len(X_test)} amostras de teste.")

    # Criar e treinar o TPOT Classifier
    logger.info("Iniciando o treinamento do modelo TPOT.")
    try:
        tpot = TPOTClassifier(
            generations=5,
            population_size=20,
            verbosity=2,
            random_state=42
        )
        tpot.fit(X_train, y_train)
        logger.info("Modelo TPOT treinado com sucesso.")
    except Exception as e:
        logger.error(f"Erro durante o treinamento do modelo TPOT: {e}")
        raise e

    # Avaliar o desempenho no conjunto de teste
    logger.info("Avaliar o desempenho no conjunto de teste.")
    score = tpot.score(X_test, y_test)
    logger.info(f"Score no conjunto de teste: {score}")

    # Exibir o pipeline otimizado encontrado
    logger.info("Exibindo o pipeline otimizado encontrado.")
    logger.info(f"Pipeline otimizado: {tpot.fitted_pipeline_}")

    # Retornar o modelo, a pontuação e o pipeline otimizado
    return tpot, score, tpot.fitted_pipeline_
