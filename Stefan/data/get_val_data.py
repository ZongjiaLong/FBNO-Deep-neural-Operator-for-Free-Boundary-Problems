import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader, random_split
from collections import namedtuple
import os
from numpy.fft import fft

from typing import NamedTuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StefanDataPoint(NamedTuple):
    """Container for Stefan problem data points."""
    mag: torch.Tensor
    period: torch.Tensor
    u: torch.Tensor
    s: torch.Tensor
    t: torch.Tensor
    phi: torch.Tensor
    x_right: torch.Tensor
    test_input: torch.Tensor


class StefanDataset(Dataset):
    """Dataset for Stefan problem simulation data.

    Args:
        data_path: Path to .npy file containing the saved data
        max_samples: Maximum number of samples to load
        num_freq: Number of frequency components (unused in current implementation)
    """

    def __init__(
            self,
            data_path: str,
            max_samples: int = 800,
            num_freq: int = 64
    ) -> None:
        raw_data = np.load(data_path, allow_pickle=True)[:max_samples]
        self.data: list[StefanDataPoint] = []
        self.num_freq = num_freq
        # factorx = 1
        # factort = 50
        for sample in raw_data:
            # Preprocess tensors
            x = torch.tensor(sample['normalized_space'], dtype=torch.float32)
            x = x.unsqueeze(1).expand(-1, 5001)  # (5001, 5001)

            t = torch.tensor(sample['normalized_time'], dtype=torch.float32)
            t = t.unsqueeze(0).expand(101, -1)  # (101, 101)

            test_input = torch.stack([x, t], dim=-1)  # (101, 50 = 5001, 2)
            # test_input = test_input[::factorx,::factort,:]
            # Create data point
            data_point = StefanDataPoint(
                mag=torch.tensor(sample['heat_magnitude'], dtype=torch.float32).unsqueeze(0),
                period=torch.tensor(sample['period'], dtype=torch.float32).unsqueeze(0),
                phi=torch.tensor(sample['physical_space'], dtype=torch.float32),
                u=torch.tensor(sample['temperature_field'], dtype=torch.float32),
                s=torch.tensor(sample['interface_position'], dtype=torch.float32),
                t=torch.tensor(sample['normalized_time'], dtype=torch.float32),
                x_right=torch.ones_like(torch.tensor(sample['normalized_time'], dtype=torch.float32)),
                test_input=test_input
            )
            self.data.append(data_point)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> StefanDataPoint:
        return self.data[idx]


def create_dataloaders(
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        max_samples: Optional[int] = None
) -> DataLoader:
    """Create DataLoader for Stefan problem dataset.

    Args:
        data_path: Path to the data file
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        DataLoader configured for the Stefan problem dataset
    """
    dataset = StefanDataset(data_path, max_samples=max_samples if max_samples is not None else 800)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
# loader = create_dataloaders("D:\\desktop\\stefan_plots\\data\\stefan_data.npy")
# for batch in loader:
#     test_input = batch.test_input.to(device='cuda')
#     print(test_input.shape)
#     # Print shapes of each component in the batch
#     print(f"Batch size: {len(batch)}")  # Number of samples in this batch
#     print(f"mag shape: {batch.mag.shape}")
#     print(f"period shape: {batch.period.shape}")
#     print(f"u shape: {batch.u.shape}")
#     print(f'phi shape: {batch.phi.shape}')
#     print(f'test_input shape: {batch.test_input.shape}')
#     print(f"x_right shape: {batch.x_right.shape}")
#     print(f"t shape: {batch.t.shape}")
#     print(f"s shape: {batch.s.shape}")
#
#
#     # Only print for the first batch to avoid too much output
#     break