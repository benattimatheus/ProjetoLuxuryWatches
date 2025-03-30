import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from models.logs.logger import logger

def preprocess_data(data: pd.DataFrame, target: str):
    logger.info("Iniciando o pré-processamento dos dados.")
    
    X = data.drop(columns=[target])
    logger.info("Coluna de destino removida do conjunto de dados.")

    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])
        logger.info("'Unnamed: 0' removida das colunas.")
    
    data['price'] = data['price'].apply(lambda x: -1 if x == 'Price on request' else x)
    data['price'] = data['price'].apply(lambda x: int(x.replace('$', '').replace(',', '').replace("'", '')) if isinstance(x, str) else -1)
    logger.info("Ajustes feitos na coluna 'price'. Valores de preço convertidos.")

    if 'size' in X.columns:
        X['size'] = X['size'].apply(lambda x: str(x).replace(' mm', '') if isinstance(x, str) else x)
        X['size'] = pd.to_numeric(X['size'], errors='coerce')
        logger.info("Coluna 'size' processada e convertida para valores numéricos.")

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    logger.info(f"Características numéricas identificadas: {numeric_features}")
    logger.info(f"Características categóricas identificadas: {categorical_features}")

    numeric_pipeline = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler()) 
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        # ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    logger.info("Pipelines de pré-processamento configurados.")

    preprocessor = ColumnTransformer(transformers=[ 
        ('num', numeric_pipeline, numeric_features), 
        ('cat', categorical_pipeline, categorical_features) 
    ])
    logger.info("ColunaTransformer configurado.")

    X_preprocessed = preprocessor.fit_transform(X)
    logger.info("Transformações realizadas no conjunto de dados.")

    # Gerar nomes das colunas transformadas
    all_columns = numeric_features + categorical_features  # Ajustar isso conforme a transformação feita
    logger.info("Nomes das colunas transformadas gerados.")

    # Converter para DataFrame para visualização
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_columns)
    logger.info("Conjunto de dados pré-processado convertido para DataFrame.")

    y = data[target]
    logger.info(f"Coluna alvo '{target}' extraída.")

    return X_preprocessed_df, y, preprocessor
