# File: src/datamodules.py
"""
Lightning DataModules for SSL pretraining and Shenzhen finetuning.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from .datasets import ChestNpzSSL, ShenzhenCSV


class ChestSSLDataModule(pl.LightningDataModule):
    """
    DataModule for DINO pretraining on ChestMNIST.
    """
    def __init__(self, npz_path, batch: int = 128, workers: int = 0):
        super().__init__()
        self.path = Path(npz_path)
        self.batch = batch
        self.workers = workers

    def setup(self, stage=None):
        self.ds = ChestNpzSSL(self.path)

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=False,
        )


class ShenzhenDM(pl.LightningDataModule):
    """
    DataModule for Shenzhen chest X-ray classification.
    Splits data into train/val/test.
    """
    def __init__(self, csv_path: Path, img_dir: Path,
                 batch: int = 32, workers: int = 0, seed: int = 42):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.batch = batch
        self.workers = workers
        self.seed = seed

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        labels = df["findings"].apply(lambda s: 0 if s.lower().strip() == "normal" else 1)
        train_idx, temp_idx = train_test_split(
            np.arange(len(df)), test_size=0.25,
            stratify=labels, random_state=self.seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.40,
            stratify=labels.iloc[temp_idx], random_state=self.seed
        )
        self.train_ds = ShenzhenCSV(df.iloc[train_idx], self.img_dir, train=True)
        self.val_ds = ShenzhenCSV(df.iloc[val_idx], self.img_dir, train=False)
        self.test_ds = ShenzhenCSV(df.iloc[test_idx], self.img_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch,
                          shuffle=True, num_workers=self.workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch,
                          shuffle=False, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch,
                          shuffle=False, num_workers=self.workers, pin_memory=True)
