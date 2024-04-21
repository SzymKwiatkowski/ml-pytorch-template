"""Module implementing class dataset"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd


# Remove when implementing dataset
# pylint: disable=W0511
class LearningDataset(Dataset):
    """Class implementing dataset for learning."""
    def __init__(self,
                 path_to_file: Path):
        super().__init__()

        self._path_to_file = path_to_file
        self._df = pd.read_csv(path_to_file)
        self.x = torch.randn(3, 520, 520)
        self.y = torch.randn(1000)

    # TODO: Return x,y of dataset
    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y

    def __len__(self) -> int:
        return 1
