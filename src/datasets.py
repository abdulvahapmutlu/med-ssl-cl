# File: src/datasets.py
"""
Dataset definitions for supervised and SSL stages.
"""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from .transforms import med_transform_ssl, DinoTransform


class ChestNpz(Dataset):
    """
    Supervised ChestMNIST loader from .npz file.
    """
    def __init__(self, npz_path: Path, split: str):
        arr = np.load(npz_path)
        if split == "test":
            self.x, self.y = arr["test_images"], arr["test_labels"]
        else:
            x, y = arr["train_images"], arr["train_labels"]
            n_val = int(0.10 * len(x))
            if split == "train":
                self.x, self.y = x[:-n_val], y[:-n_val]
            else:
                self.x, self.y = x[-n_val:], y[-n_val:]
        self.tf = med_transform_ssl(gray=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = (self.x[idx] * 255).astype("uint8")
        img = Image.fromarray(img).convert("L")
        img = self.tf(img).repeat(3, 1, 1)
        label = torch.from_numpy(self.y[idx]).float()
        return img, label


class ChestNpzSSL(Dataset):
    """
    SSL-only loader for ChestMNIST using DinoTransform.
    """
    def __init__(self, npz_path: Path):
        arr = np.load(npz_path)
        self.data = arr["train_images"]
        self.tf = DinoTransform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = (self.data[idx] * 255).astype("uint8")
        img = Image.fromarray(img).convert("L")
        return self.tf(img)


class ShenzhenCSV(Dataset):
    """
    Supervised loader for Shenzhen CXR from CSV metadata.
    """
    def __init__(self, df, img_dir: Path, train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        mean = std = [0.5]
        ops = [transforms.Resize((224, 224))]
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        self.tf = transforms.Compose(ops)

    @staticmethod
    def _label(findings: str) -> int:
        return 0 if findings.lower().strip() == "normal" else 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["study_id"]).convert("L")
        x = self.tf(img).repeat(3, 1, 1)
        y = torch.tensor(self._label(row["findings"]), dtype=torch.float32)
        return x, y


class CXRCSV(ShenzhenCSV):
    """
    Generic CXR loader for binary classification.
    Inherits same behavior as ShenzhenCSV.
    """
    pass


class MIASCSV(Dataset):
    """
    Supervised loader for MIAS mammogram dataset.
    """
    def __init__(self, df, img_dir: Path, train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        # SEVERITY: malignant if startswith 'M'
        sev = self.df["SEVERITY"].fillna("B").astype(str).str.upper()
        self.targets = [1 if s.startswith("M") else 0 for s in sev]
        # for oversampling
        self.pos = [i for i, t in enumerate(self.targets) if t == 1]
        self.neg = [i for i, t in enumerate(self.targets) if t == 0]
        ops = [transforms.Resize(256), transforms.CenterCrop(224)]
        if train:
            ops = [transforms.Resize(256),
                   transforms.RandomResizedCrop(224, scale=(0.85,1.0), ratio=(0.9,1.1)),
                   transforms.RandomHorizontalFlip(),
                   transforms.RandomRotation(5)]
        ops += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        stem = self.df.iloc[idx]["REFNUM"]
        # find file with .png or .pgm
        for ext in (".png", ".pgm"):
            path = self.img_dir / f"{stem}{ext}"
            if path.exists():
                img = Image.open(path).convert("L")
                break
        x = self.tf(img).repeat(3, 1, 1)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y
