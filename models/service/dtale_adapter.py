# adapters/dtale_adapter.py

from ports.dtale_port import DtalePort
import dtale
import pandas as pd

class DtaleAdapter(DtalePort):
    def open_in_dtale(self, df: pd.DataFrame) -> pd.DataFrame:
        d = dtale.show(
            df,
            subprocess=False,
            host="localhost",
            port=40000,
            open_browser=False
        )

        print(f"Dtale está rodando em: {d._main_url}")
        print("Abra essa URL no navegador para editar os dados.")
        input("Pressione Enter quando terminar de editar no D-Tale...")

        try:
            # Tenta pegar os dados modificados
            edited_df = d.data
            print("Dados editados recuperados com sucesso!")
            return edited_df
        except Exception as e:
            print("Erro ao recuperar dados editados. Retornando original.")
            print(f"Erro: {e}")
            return df
