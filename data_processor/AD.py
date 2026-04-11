# LaBraM/data_processor/AD.py
import h5py
import numpy as np
from torch.utils.data import Dataset

class ADDataset(Dataset):
    """
    Loads windowed EEG + binary AD/control labels from H5 files
    produced by make_ad_dataset.py.

    Expected H5 structure:
        X: float32 (n_windows, n_channels, n_times)
        y: int64   (n_windows,)
    """
    def __init__(self, data_path: str, channel_names: list[str]):
        super().__init__()
        with h5py.File(data_path, "r") as f:
            self.X = f["X"][:]   # load fully into RAM; use h5py lazy load if memory-constrained
            self.y = f["y"][:]
        self.channel_names = channel_names

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # LaBraM expects (n_channels, n_patches, patch_size)
        # Your X is (n_channels, n_times) where n_times = 800 (4s @ 200Hz)
        # patch_size=200, so n_patches = 800/200 = 4
        x = self.X[idx] # (n_channels, 800) — leave flat, engine will reshape
        return x, self.y[idx]
