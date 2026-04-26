import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, shuffle=False, batch_size=64):

    t = lambda a: torch.tensor(a.values if hasattr(a, "values") else a,
                               dtype=torch.float32).to(DEVICE)

    X_tensor = t(X)
    y_tensor = t(y).view(-1, 1)  # importante para BCELoss
    tensor_ds = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(tensor_ds, batch_size=batch_size, shuffle=shuffle)

    return loader