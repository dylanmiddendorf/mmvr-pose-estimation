import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data as data
from torch import Tensor


class MMVR(data.Dataset):
    def __init__(
        self,
        X: npt.NDArray[np.float16],
        y: npt.NDArray[np.float16],
        kp: npt.NDArray[np.float16],
    ):
        assert len(X) == len(y)

        self.X = torch.tensor(X, dtype=torch.float16)
        self.y = torch.tensor(y, dtype=torch.float16)
        self.kp = torch.tensor(kp, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.X[idx], self.y[idx], self.kp[idx]
