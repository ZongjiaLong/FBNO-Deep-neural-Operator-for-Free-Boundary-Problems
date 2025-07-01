import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader, random_split
from collections import namedtuple
import os
from numpy.fft import fft
from matplotlib.path import Path
# Define a named tuple for our data points
ContourDataPoint = namedtuple('ContourDataPoint',
                              [ 'fx',  'coord', 'value','initial_domain'])




class ContourDataset(Dataset):
    def __init__(self, data_path, max_samples=800):
        """
        Args:
            data_path (string): Path to the .npy file containing the saved data
            max_samples (int): Maximum number of samples to load (default: 800)
        """
        raw_data = np.load(data_path, allow_pickle=True)[:max_samples]
        self.data = []

        for sample in raw_data:
            contour = sample['coord']
            contour_tensor = torch.tensor(contour, dtype=torch.float32)
            self.data.append(ContourDataPoint(
                fx=torch.tensor(sample['s1'], dtype=torch.float32).unsqueeze(0),
                value=torch.tensor(sample['value'], dtype=torch.float32),
                coord=contour_tensor,
                initial_domain=torch.tensor(sample['initial_domain'], dtype=torch.float32).unsqueeze(0),

            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def create_dataloaders(data_path, batch_size=32, shuffle=True, max_samples=800,num_random_points=500
                       ):

    full_dataset = ContourDataset(data_path, max_samples=max_samples)


    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader
# test_path = "F:\\Tumour\\picture\\boundary_data\\train_with_four\\domain_test.npy"
#
# test_loader = create_dataloaders(test_path, batch_size=10,shuffle = False, max_samples=200)

# for batch in test_loader:
#     print(f"Batch s1 shape: {batch.s1.shape}")
#     print(f"Batch coord shape: {batch.coord.shape}")
#     print(f"Batch value shape: {batch.value.shape}")
#     print(f"Batch initial_domain shape: {batch.initial_domain.shape}")
#     break