import shap
from models.logs.logger import logger

def calculate_shap_values(model, data, target: str):
    """
    Calcula os valores SHAP para as features usando o modelo treinado.
    
    Parâmetros:
      - model: Modelo treinado (deve ter um método predict ou similar).
      - data: DataFrame com os dados completos.
      - target: Nome da coluna alvo (a ser removida para obter X).
    
    Retorna:
      - Objeto com os valores SHAP.
    """
    logger.info(f"Iniciando o cálculo dos valores SHAP para o modelo com a variável alvo '{target}'.")

    # Separa as features (X) do target (y)
    X = data.drop(columns=[target])

    logger.info(f"Shape de X para cálculo dos valores SHAP: {X.shape}")

    try:
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        logger.info("Cálculo dos valores SHAP concluído com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao calcular os valores SHAP: {e}")
        shap_values = None
    
    return shap_values
