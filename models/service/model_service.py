from model_adapter import train_model_sklearn
from models.logs.logger import logger

def build_model(data, target: str):
    """
    Orquestra o treinamento do modelo.
    """
    logger.info(f"Iniciando a construção do modelo para a variável alvo '{target}'.")

    model, le = train_model_sklearn(data, target)

    if model is not None:
        logger.info(f"Modelo treinado com sucesso: {model}")
    else:
        logger.warning("Falha no treinamento do modelo. Nenhum modelo retornado.")

    return model, le
