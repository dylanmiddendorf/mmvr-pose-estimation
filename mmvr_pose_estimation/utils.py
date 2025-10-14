from typing import Optional, Callable
import torch
from torch import Tensor
import torch.utils.data as data


class MMVR(data.Dataset):
    def __init__(self, X, y, transform: Optional[Callable[[Tensor], Tensor]] = None):
        assert len(X) == len(y)

        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_i, y_i = self.X[idx], self.y[idx]

        if self.transform is not None:
            X_i = self.transform(X_i)
        return X_i, y_i
