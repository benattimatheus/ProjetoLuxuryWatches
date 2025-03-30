import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from models.logs.logger import logger

def train_model_sklearn(data: pd.DataFrame, target: str):
    logger.info("Iniciando o treinamento do modelo.")
    
    X = data.drop(columns=[target])
    y = data[target]

    if y.dtype == 'object' or y.dtype.name == 'category':
        logger.info("Coluna alvo é categórica. Aplicando LabelEncoder.")
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None
        logger.info("Coluna alvo é numérica. Nenhuma codificação necessária.")
    
    logger.info(f"Shape de X: {X.shape}")
    logger.info(f"Shape de y: {y.shape}")

    numeric_columns = X.select_dtypes(include=[np.number])
    if numeric_columns.empty:
        logger.warning("Nenhuma coluna numérica encontrada em X. Verifique o pré-processamento dos dados.")
        return None, None  # Retorna None para ambos os valores
    
    logger.info(f"Shape de X após o pré-processamento: {X.shape}")

    if 'size' not in X.columns:
        logger.warning("Coluna 'size' não encontrada. Renomeando a coluna 0 para 'size'.")
        X.columns = ['size']  # Renomear a coluna numerada para 'size'

    logger.info("Primeiros 5 valores da coluna 'size' antes da conversão:")
    logger.info(f"{X['size'].head()}")

    X['size'] = pd.to_numeric(X['size'], errors='coerce')  # Tentar converter explicitamente a coluna 'size'
    
    logger.info("Primeiros 5 valores da coluna 'size' após conversão:")
    logger.info(f"{X['size'].head()}")

    logger.info(f"Número de valores NaN após conversão: {X['size'].isna().sum()}")

    numeric_columns = X.select_dtypes(include=[np.number])
    logger.info(f"Colunas numéricas após conversão: {numeric_columns.columns.tolist()}")
    
    if numeric_columns.empty:
        logger.warning("Nenhuma coluna numérica encontrada após conversão. Verifique o pré-processamento dos dados.")
        return None, None  # Retorna None se não houver colunas numéricas

    logger.info(f"Shape de X após conversão para numérico: {X.shape}")

    candidates = {
        'gradient_boosting': {
            'pipeline': Pipeline([('scaler', StandardScaler()), 
                                  ('regressor', GradientBoostingRegressor(random_state=123))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7]
            }
        },
        'ada_boost': {
            'pipeline': Pipeline([('regressor', AdaBoostRegressor(random_state=123))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'xgboost': {
            'pipeline': Pipeline([('regressor', xgb.XGBRegressor(random_state=123, objective='reg:squarederror'))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7]
            }
        },
        'lightgbm': {
            'pipeline': Pipeline([('regressor', LGBMRegressor(random_state=123))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [-1, 5, 10]
            }
        }
    }

    best_score = -np.inf
    best_model = None
    best_model_name = None

    for name, candidate in candidates.items():
        logger.info(f"Iniciando treinamento do modelo: {name}")
        grid = GridSearchCV(candidate['pipeline'], candidate['params'], cv=5, scoring='r2', n_jobs=-1)
        
        logger.info(f"Shape de X antes do treinamento: {X.shape}")
        logger.info(f"Shape de y antes do treinamento: {y.shape}")
        
        try:
            grid.fit(X, y)
            logger.info(f"Modelo: {name} | Melhor R² CV: {grid.best_score_:.4f}")
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                best_model_name = name
        except Exception as e:
            logger.error(f"Erro com o modelo {name}: {e}")

    logger.info(f"\nModelo Selecionado: {best_model_name} com R² CV: {best_score:.4f}")
    
    return best_model, le
