"""Module implementing class dataset"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np


# Remove when implementing dataset
# pylint: disable=W0511
class LearningDataset(Dataset):
    """Class implementing dataset for learning."""
    def __init__(self):
        super().__init__()
        
        self.arr = np.array(np.random.rand(100, 3, 520, 520), dtype=np.float32)
        self.y = np.array(np.random.rand(100, 1000), dtype=np.float32)

    # TODO: Return x,y of dataset
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.arr[i]), torch.from_numpy(self.y[i])

    def __len__(self) -> int:
        return len(self.arr)
