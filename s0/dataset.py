"""
dataset.py - PyTorch Dataset for S0 data.

Purpose:
  Provide PyTorch-friendly access to the separated tensor outputs.
  Each sample returns continuous, interventions, proxy, all masks, and label.

Connects to:
  - data/s0/processed/ for tensors
  - data/s0/splits.json for indices

How to run:
  from s0.dataset import S0Dataset, build_s0_dataloaders
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class S0Dataset(Dataset):
    """
    PyTorch Dataset for S0 processed outputs.

    Each sample is a dict:
      continuous:           (T, Fc) float32
      interventions:        (T, Fi) float32
      proxy:                (T, Fp) float32
      mask_continuous:      (T, Fc) float32
      mask_interventions:   (T, Fi) float32
      mask_proxy:           (T, Fp) float32
      label:                scalar float32 (mortality)
      center_id:            str
    """

    def __init__(
        self,
        proc_dir: Path,
        static_path: Path,
        indices: np.ndarray | None = None,
        label_col: str = "mortality_inhospital",
    ):
        proc_dir = Path(proc_dir)

        self.continuous = np.load(proc_dir / "continuous.npy")
        self.interventions = np.load(proc_dir / "interventions.npy")
        self.proxy = np.load(proc_dir / "proxy_indicators.npy")
        self.mask_cont = np.load(proc_dir / "masks_continuous.npy")
        self.mask_int = np.load(proc_dir / "masks_interventions.npy")
        self.mask_proxy = np.load(proc_dir / "masks_proxy.npy")

        static = pd.read_csv(static_path)
        self.labels = static[label_col].fillna(0).values.astype(np.float32)
        self.center_ids = static["center_id"].values

        if indices is not None:
            self.indices = np.array(indices)
        else:
            self.indices = np.arange(len(self.labels))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        return {
            "continuous": torch.from_numpy(self.continuous[i]).float(),
            "interventions": torch.from_numpy(self.interventions[i]).float(),
            "proxy": torch.from_numpy(self.proxy[i]).float(),
            "mask_continuous": torch.from_numpy(self.mask_cont[i]).float(),
            "mask_interventions": torch.from_numpy(self.mask_int[i]).float(),
            "mask_proxy": torch.from_numpy(self.mask_proxy[i]).float(),
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
            "center_id": self.center_ids[i],
        }


def build_s0_dataloaders(
    s0_dir: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    label_col: str = "mortality_inhospital",
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from S0 outputs."""
    s0_dir = Path(s0_dir)
    proc_dir = s0_dir / "processed"
    static_path = s0_dir / "static.csv"
    splits_path = s0_dir / "splits.json"

    with open(splits_path) as f:
        splits = json.load(f)

    loaders = {}
    for split_name in ["train", "val", "test"]:
        indices = np.array(splits[split_name])
        ds = S0Dataset(proc_dir, static_path, indices, label_col)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            drop_last=(split_name == "train"),
        )

    return loaders
