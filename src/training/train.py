import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
print(f"SRC_DIR: {SRC_DIR}")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ----------------------------------------------------------- #

# Imports internos
# Imports externos
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from enums.dataset_type import DatasetType
from model.architecture import ChurnMLP, EarlyStopping
from pipeline.builder import PipelineBuilder
from services.dataframe_service import DataFrameService
from services.mlflow_service import MLFlowService
from services.preprocessing_service import PreprocessingService
from utils.feature_identifier import FeatureIdentifier
from utils.loaders import make_loader

# Carregamento de varaveis e instanciação de serviços

load_dotenv()

df_service  = DataFrameService() # Carregamento dos Dados
pipeline_builder = PipelineBuilder() # Criação da pipeline de preprocessamento
feature_identifier = FeatureIdentifier() # Identificação das features (categoricas e textuais)
preprocessing_service  = PreprocessingService(pipeline_builder=pipeline_builder, feature_identifier=feature_identifier)
mlflow_service = MLFlowService(experiment_name="churn_prediction")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Parametros usados no treinamento do modelo
EPOCHS = 100
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-3
PATIENCE = 10
DROPOUT = 0.3

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts")

log_params = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "weight_decay": WEIGHT_DECAY,
    "learning_rate": LEARNING_RATE,
    "patience": PATIENCE,
    "dropout": DROPOUT
}

import joblib


def preprocessing() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    # Carregameno dos dados
    _default_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = df_service.load_dataframe(os.getenv("PREPROCESSING_FILE_PATH", _default_path))
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])

    # Separação em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Separação do treino em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Executa pipeline de preprocessamento de treino (fit + transform)
    X_train_proc = preprocessing_service.run_pipeline(X_train, type=DatasetType.TRAIN)

    # Executa pipeline de preprocessamento de validação (transfom)
    X_val_proc  = preprocessing_service.run_pipeline(X_val,  type=DatasetType.VALIDATION)

    # Salva pipeline ajustado para uso em produção (API de inferência)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    pipeline_path = os.path.join(ARTIFACTS_DIR, "pipeline.pkl")
    joblib.dump(preprocessing_service.pipeline, pipeline_path)
    print(f"Pipeline salvo em: {pipeline_path}")

    return X_train_proc, y_train, X_val_proc, y_val

def train_model(X_train, y_train, X_val, y_val, epochs=EPOCHS):

    # Instaciação do Modelo
    num_cols = X_train.shape[1]
    model = ChurnMLP(input_dim=num_cols, dropout=0.3).to(DEVICE)

    # Criação de loaders 
    train_loader = make_loader(X_train, y_train, shuffle=True)
    validation_loader = make_loader(X_val, y_val,shuffle=False)

    # Peso do erro
    positive_weight_val = (y_train == 0).sum() / (y_train == 1).sum()
    pos_weight = torch.tensor([positive_weight_val], dtype=torch.float32).to(DEVICE)

    # Loss Function
    criterion = nn.BCEWithLogitsLoss(weight=pos_weight)

    # Otimizador
    weight_decay = WEIGHT_DECAY
    learning_rate = LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)

    # Instancia Early Stopping
    early_stopping = EarlyStopping(patience = PATIENCE)
    best_state = None  # rastreia melhor estado fora do loop

    # Inicia Run
    mlflow_service.start_run(run_name="train_model")
    mlflow_service.log_params(log_params)

    for epoch in range(1, epochs + 1):
        # — Treino —

        model.train()
        train_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()

            pred = model(xb)
            yb = yb.view(-1)
            loss = criterion(pred, yb)
            loss.backward()

            # Gradient clipping: evita explosão de gradientes
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)


        # Validação
        model.eval()
        val_losses = []


        with torch.no_grad():
            for xb, yb in validation_loader:
                pred = model(xb)
                yb = yb.view(-1)  # reshape para (batch_size,)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)

        # Log epoch metrics
        mlflow_service.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            step=epoch,
        )

        # mlflow_service.log_pytorch_model(model, name=f"checkpoint_{epoch}")

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early Stopping
        if early_stopping.step(val_loss, model):
            print(f"Early stopping na epoca {epoch}")
            best_state = {k: v.clone() for k, v in early_stopping.best_model.items()}
            break
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    mlflow_service.log_pytorch_model(model, name="final_model", export_model=False)

    return model

def save_model(model):

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)  # Garante que o diretório existe
    model_path = os.path.join(ARTIFACTS_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em: {model_path}")

def main():
    X_train, y_train, X_val_, y_val  = preprocessing()
    
    model = train_model(X_train, y_train, X_val_, y_val)
    save_model(model)

if __name__ == "__main__":
    main()