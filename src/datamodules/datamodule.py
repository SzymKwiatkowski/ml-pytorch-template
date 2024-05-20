"""Datamodule implemention for learning dataset"""
from pathlib import Path
from typing import Optional

import pandas as pd
from lightning import pytorch as pl
from torch.utils.data import DataLoader

from datamodules.dataset_split import basic_split
from datasets.dataset import LearningDataset


class DataModule(pl.LightningDataModule):
    """Class implementation for datamodule for learning dataset"""
    def __init__(
            self,
            data_path: Path,
            batch_size: int = 32,
            num_workers: int = 4,
            train_size: float = 0.8):
        super().__init__()

        self._data_path = Path(data_path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_size = train_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.train_dataset = LearningDataset()

        self.val_dataset = LearningDataset()

        self.test_dataset = LearningDataset()

        self.save_hyperparameters(ignore=['data_path', 'number_of_workers'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )
