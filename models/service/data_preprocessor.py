import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from models.logs.logger import logger


def preprocess_data(data: pd.DataFrame, target: str):
    logger.info("Iniciando o pré-processamento dos dados.")
    
    # Corrige o valor do preço
    data['price'] = data['price'].apply(lambda x: -1 if x == 'Price on request' else x)
    data['price'] = data['price'].apply(lambda x: int(x.replace('$', '').replace(',', '').replace("'", '')) if isinstance(x, str) else x)
    logger.info("Ajustes feitos na coluna 'price'. Valores de preço convertidos.")

    # Remove linhas com price == -1
# Corrige o valor do preço
    data['price'] = data['price'].apply(lambda x: -1 if x == 'Price on request' else x)
    data['price'] = data['price'].apply(lambda x: int(x.replace('$', '').replace(',', '').replace("'", '')) if isinstance(x, str) else x)
    data[target] = pd.to_numeric(data[target], errors='coerce')  # Garante que é numérico e converte inválidos para NaN

# Remove valores inválidos (NaN ou -1)
    data = data[~data[target].isna()]  # remove NaN
    data = data[data[target] != -1]    # remove -1
    logger.info("Linhas com valores ausentes ou -1 removidas da coluna alvo.")


    X = data.drop(columns=[target])
    y = data[target]

    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])
        logger.info("'Unnamed: 0' removida das colunas.")

    # Preprocessar size se existir
    if 'size' in X.columns:
        X['size'] = X['size'].apply(lambda x: str(x).replace(' mm', '') if isinstance(x, str) else x)
        X['size'] = pd.to_numeric(X['size'], errors='coerce')
        logger.info("Coluna 'size' processada e convertida para valores numéricos.")

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    logger.info(f"Características numéricas: {numeric_features}")
    logger.info(f"Características categóricas: {categorical_features}")

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Cria um pipeline para aplicar TargetEncoder
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_encoder', TargetEncoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    logger.info("Preprocessor com TargetEncoder montado com sucesso.")

    X_preprocessed = preprocessor.fit_transform(X, y)
    logger.info("Pré-processamento aplicado.")

    # Após target encoding, as categorias viram numéricas
    processed_columns = numeric_features + categorical_features
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=processed_columns)
    logger.info("Pré-processado convertido para DataFrame.")

    return X_preprocessed_df, y, preprocessor
