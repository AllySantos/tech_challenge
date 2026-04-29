from pathlib import Path

import pandas as pd


class DataFrameService:
    def __init__(self):
        pass

    def load_dataframe(self, path: str) -> pd.DataFrame:
        """Carrega um DataFrame a partir de um arquivo CSV."""
        try:
            # Resolve o caminho relativo a partir do diretório onde este arquivo está
            current_dir = Path(__file__).parent.parent.parent  # tech_challenge
            file_path = current_dir / path if not Path(path).is_absolute() else Path(path)

            df = pd.read_csv(file_path)
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
            return df
        except Exception as e:
            print(f"Erro ao carregar o DataFrame: {e}")
            raise
