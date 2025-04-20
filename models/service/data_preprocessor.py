import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from models.logs.logger import logger

def preprocess_data(data: pd.DataFrame, target: str):
    logger.info("Iniciando o pré-processamento dos dados.")
    
    # Corrigir preço
    data['price'] = data['price'].apply(lambda x: -1 if x == 'Price on request' else x)
    data['price'] = data['price'].apply(lambda x: int(x.replace('$', '').replace(',', '').replace("'", '')) if isinstance(x, str) else x)
    data[target] = pd.to_numeric(data[target], errors='coerce')
    
    # Remover valores ausentes ou inválidos
    data = data[~data[target].isna()]
    data = data[data[target] != -1]
    logger.info("Linhas com valores ausentes ou -1 removidas da coluna alvo.")

    # Remover colunas pouco informativas ou muito específicas
    drop_cols = ['Unnamed: 0', 'name', 'ref']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns], errors='ignore')

    # Criar novas features
    if 'yop' in data.columns:
        # Tenta extrair o ano (os 4 primeiros dígitos)
        data['yop'] = data['yop'].astype(str).str.extract(r'(\d{4})')  # pega apenas o ano
        data['yop'] = pd.to_numeric(data['yop'], errors='coerce')  # converte para numérico

        current_year = pd.Timestamp.now().year
        data['watch_age'] = current_year - data['yop']
        logger.info("Coluna 'yop' limpa e feature 'watch_age' criada.")


    if 'casem' in data.columns and 'bracem' in data.columns:
        data['has_gold'] = data[['casem', 'bracem']].apply(lambda x: int('gold' in ' '.join(map(str, x)).lower()), axis=1)
        logger.info("Feature binária 'has_gold' criada com base em materiais.")

    if 'size' in data.columns:
        data['size'] = data['size'].apply(lambda x: str(x).replace(' mm', '') if isinstance(x, str) else x)
        data['size'] = pd.to_numeric(data['size'], errors='coerce')
        logger.info("Coluna 'size' convertida para numérico.")

    # Reduzir cardinalidade de 'model' se necessário
    if 'model' in data.columns:
        top_models = data['model'].value_counts().nlargest(30).index
        data['model'] = data['model'].apply(lambda x: x if x in top_models else 'Other')
        logger.info("Cardinalidade da coluna 'model' reduzida (top 30).")

    # Aplicar log1p ao target (preço)
    data[target] = data[target].apply(lambda x: np.log1p(x) if x > 0 else 0)
    logger.info("Transformação log1p aplicada ao alvo.")

    # Separar X e y
    logger.info(f"Shape após limpeza inicial: {data.shape}")

    # Separar X e y
    X = data.drop(columns=[target])
    y = data[target]
    
    # Features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numéricas: {numeric_features}")
    logger.info(f"Categóricas: {categorical_features}")
    logger.info(f"Nº de amostras: {len(X)}")
    
    # Evitar erro se tiver poucas amostras
    if len(X) < 5:
        logger.warning("Número muito pequeno de amostras para modelagem!")
    
    # Pipelines
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_encoder', TargetEncoder(smoothing=0.3))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Aplicar preprocessor
    logger.info("Aplicando pré-processador...")
    X_preprocessed = preprocessor.fit_transform(X, y)
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=numeric_features + categorical_features)
    
    # Verificar variância das colunas
    for col in X_preprocessed_df.columns:
        nunique = X_preprocessed_df[col].nunique()
        logger.info(f"Coluna {col} - valores únicos: {nunique}")
    
    # Validar se ainda tem dados
    if X_preprocessed_df.empty or y.empty:
        logger.error("Dataset vazio após pré-processamento.")
        raise ValueError("Dataset vazio após pré-processamento.")
    
    logger.info(f"Shape final de X: {X_preprocessed_df.shape}")
    logger.info(f"Pré-processamento finalizado com sucesso.")
    return X_preprocessed_df, y, preprocessor