import pandas as pd
import sweetviz as sv
import dtale
from autoviz.AutoViz_Class import AutoViz_Class
from ydata_profiling import ProfileReport
import logging
from models.logs.logger import logger


logger = logging.getLogger("EDA_Project")

class EDAReport:
    def __init__(self, dataset_path: str):
        try:
            self.dataset = pd.read_csv(dataset_path)
            logger.info(f"Dataset carregado com sucesso: {dataset_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar o dataset: {e}")
            self.dataset = None

    def generate_autoviz(self):
        if self.dataset is not None:
            av = AutoViz_Class()
            av.AutoViz(self.dataset)
        else:
            logger.error("Dataset não carregado. Relatório AutoViz não pode ser gerado.")

    def generate_sweetviz(self):
        if self.dataset is not None:
            report = sv.analyze(self.dataset)
            report.show_html("sweetviz_report.html")
        else:
            logger.error("Dataset não carregado. Relatório Sweetviz não pode ser gerado.")

    def generate_dtale(self):
        if self.dataset is not None:
            dtale.show(self.dataset)
        else:
            logger.error("Dataset não carregado. Relatório D-Tale não pode ser gerado.")

    def generate_ydata(self):
        if self.dataset is not None:
            profile = ProfileReport(self.dataset, explorative=True)
            profile.to_file("views/ydata_profiling_report.html")
            logger.info("Relatório ydata-profiling gerado com sucesso.")
        else:
            logger.error("Dataset não carregado. Relatório ydata-profiling não pode ser gerado.")


