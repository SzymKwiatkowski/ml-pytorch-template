from pathlib import Path
from typing import Optional

import pandas as pd
from lightning import pytorch as pl
from torch.utils.data import DataLoader

from datamodules.dataset_split import DatasetSplits
from datasets.controller import LearningDataset


class DataModule(pl.LightningDataModule):
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

        self.save_hyperparameters(ignore=['data_path', 'number_of_workers'])

    def setup(self, stage: Optional[str] = None):
        # Split if needed
        df = pd.read_csv(self._data_path / "train.csv")
        train_places_dirs, val_places_dirs = DatasetSplits.basic_split(df, self._train_size)

        print(f'Number of train places: {len(train_places_dirs)}')
        print(f'Number of val places: {len(val_places_dirs)}')

        self.train_dataset = LearningDataset(
            self._data_path / 'train.csv'
        )

        self.val_dataset = LearningDataset(
            self._data_path / 'val.csv'
        )

        self.test_dataset = LearningDataset(
            self._data_path / 'test.csv'
        )

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
