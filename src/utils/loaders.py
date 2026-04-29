import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model.architecture import ChurnMLP

ARTIFACTS_DIR = Path(__file__).parent.parent  /  "model" / "artifacts"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, shuffle=False, batch_size=64):

    t = lambda a: torch.tensor(a.values if hasattr(a, "values") else a,
                               dtype=torch.float32).to(DEVICE)

    X_tensor = t(X)
    y_tensor = t(y).view(-1, 1)  # importante para BCELoss
    tensor_ds = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(tensor_ds, batch_size=batch_size, shuffle=shuffle)

    return loader



def load_model(input_dim: int, checkpoint_name: str = "best_model.pt") -> ChurnMLP:
    """
    Carrega os pesos salvos em model/artifacts/<checkpoint_name>.
    O modelo é colocado em eval() — BatchNorm e Dropout se comportam
    de forma determinística para inferência.
    """
    path = ARTIFACTS_DIR / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado em {path}. "
            "Execute training/train.py antes de iniciar a API."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ChurnMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_scaler(scaler_name: str = "scaler.pkl"):
    """Carrega o StandardScaler serializado durante o treino."""
    path = ARTIFACTS_DIR / scaler_name
    if not path.exists():
        raise FileNotFoundError(f"Scaler não encontrado em {path}.")
    with open(path, "rb") as f:
        return pickle.load(f)
