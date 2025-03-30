from powershap_adapter import calculate_shap_values
from models.logs.logger import logger

def analyze_features(model, data, target: str):
    """
    Analisa as features utilizando SHAP e retorna um resumo dos valores.
    """
    logger.info(f"Iniciando a análise das features para o modelo com a variável alvo '{target}'.")

    shap_summary = calculate_shap_values(model, data, target)
    logger.info(f"Análise SHAP concluída para o modelo com a variável alvo '{target}'.")

    return shap_summary
