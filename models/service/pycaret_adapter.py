# adapters/pycaret_adapter.py
import pandas as pd
import matplotlib.pyplot as plt
from ports.training_port import TrainingPort
from models.logs.logger import logger

# PyCaret tasks
from pycaret.classification import (
    setup as class_setup,
    compare_models as class_compare,
    plot_model as class_plot,
    predict_model as class_predict
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    plot_model as reg_plot,
    predict_model as reg_predict
)
from pycaret.clustering import (
    setup as clus_setup,
    create_model as clus_create,
    plot_model as clus_plot
)

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import json
import shap
import numpy as np
import os

def save_model_metrics(y_true, y_pred, path):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred)
    }
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)


class PyCaretAdapter(TrainingPort):
    def train_model(self, df: pd.DataFrame, target: str, task_type: str):
        """
        Use PyCaret to train. We'll just return the best model object.
        """
        logger.info(f"Iniciando setup do PyCaret para tarefa: {task_type}")
        #df = df.dropna(subset=[target])
        df = df.reset_index(drop=True)

        model = None
        plot_func = None
        preds = None

        if task_type == "classification":
            class_setup(data=df, target=target, session_id=123, html=False, silent=True)
            model = class_compare()
            preds = class_predict(model, data=df)
            plot_func = class_plot
            logger.info("Modelo de classificação treinado com sucesso.")

        elif task_type == "regression":
            reg_setup(
                data=df,
                target=target,
                session_id=123,
                html=False,
                normalize=True,
                polynomial_features=True,
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                use_gpu=True
            )
            model = reg_compare()
            preds = reg_predict(model, data=df)
            plot_func = reg_plot
            logger.info("Modelo de regressão treinado com sucesso.")

        elif task_type == "clustering":
            clus_setup(data=df, session_id=123, html=False)
            model = clus_create("kmeans")
            plot_func = clus_plot
            logger.info("Modelo de clustering criado com sucesso.")
        else:
            logger.error(f"Tarefa inválida passada para treinamento: {task_type}")
            raise ValueError("Invalid task_type. Choose classification, regression, or clustering.")

        # Salvar gráfico do modelo
        try:
            logger.info("Gerando gráfico do modelo...")
        
            plot_type = 'confusion_matrix' if task_type == "classification" else 'residuals'
            plot_func(model, plot=plot_type, save=True)
        
            model_plot_path = f"views/{plot_type}.png"
            logger.info(f"Gráfico do modelo salvo automaticamente como {model_plot_path}")
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico do modelo: {e}")


        # Salvar métricas se for regressão ou classificação
        if task_type in ["classification", "regression"]:
            try:
                metrics_path = f"views/{task_type}_metrics.json"
                save_model_metrics(df[target], preds['Label'], metrics_path)
                logger.info(f"Métricas salvas em {metrics_path}")
            except Exception as e:
                logger.error(f"Erro ao salvar métricas: {e}")

            # SHAP
            try:
                logger.info("Gerando gráfico SHAP...")
                explainer = shap.Explainer(model)
                shap_values = explainer(df.drop(columns=[target]))
                shap.summary_plot(shap_values, df.drop(columns=[target]), show=False)
                shap_path = f"views/{task_type}_shap_summary.png"
                plt.savefig(shap_path)
                plt.clf()
                logger.info(f"Gráfico SHAP salvo em {shap_path}")
            except Exception as e:
                logger.error(f"Erro ao gerar gráfico SHAP: {e}")

        print(f"Best {task_type.capitalize()} Model:", model)
        return model
