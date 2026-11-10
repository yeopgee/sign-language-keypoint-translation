"""Load prepared keypoint sequences and token IDs from matching NumPy parts."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PAD_ID = 3


class KeypointTextDataset(Dataset):
    def __init__(self, directory):
        directory = Path(directory)
        self.samples = []
        x_paths = sorted(directory.glob('X_part_*.npy'))
        if not x_paths:
            raise FileNotFoundError(f'No X_part_*.npy files in {directory}')
        for x_path in x_paths:
            y_path = directory / x_path.name.replace('X_part_', 'y_part_')
            if not y_path.exists():
                raise FileNotFoundError(y_path)
            xs = np.load(x_path, allow_pickle=True)
            ys = np.load(y_path, allow_pickle=True)
            if len(xs) != len(ys):
                raise ValueError(f'Mismatched X and y lengths in {directory}')
            self.samples.extend(zip(xs, ys))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]
        return torch.as_tensor(np.asarray(x, dtype=np.float32)), torch.as_tensor(np.asarray(y, dtype=np.int64))


def collate(batch, max_frames=100):
    xs, ys = zip(*batch)
    frames = min(max(len(x) for x in xs), max_frames)
    x_batch = torch.stack([F.pad(x[:frames], (0, 0, 0, frames - min(len(x), frames))) for x in xs])
    tokens = max(len(y) for y in ys)
    y_batch = torch.stack([F.pad(y, (0, tokens - len(y)), value=PAD_ID) for y in ys])
    return x_batch, y_batch


def make_loader(directory, batch_size=64, shuffle=False, num_workers=0):
    return DataLoader(KeypointTextDataset(directory), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, collate_fn=collate)
