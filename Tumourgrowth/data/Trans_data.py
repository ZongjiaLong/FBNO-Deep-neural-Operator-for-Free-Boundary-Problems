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
                              ['step', 'fx', 'processed_contour', 'circle_points', 'feq','initial_domain'])


def even_circle_points(batch_size,t):
    """Generate evenly spaced circle points starting from positive x-axis moving anti-clockwise"""
    theta = torch.linspace(0, 2 * math.pi, batch_size + 1)[:-1]  # exclude last point to avoid duplicate
    r = 1  # radius
    t = t.unsqueeze(1).expand(-1,  batch_size).unsqueeze(2)
    points = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=1).unsqueeze(0).repeat( 15,1,1)
    points = torch.cat((points,t), dim=2)
    points = points.reshape(points.shape[0]*points.shape[1], points.shape[2])
    return points


class ContourDataset(Dataset):
    def __init__(self, data_path, max_samples=800, num_freq=64,num_random_points=3000):
        """
        Args:
            data_path (string): Path to the .npy file containing the saved data
            max_samples (int): Maximum number of samples to load (default: 800)
        """
        raw_data = np.load(data_path, allow_pickle=True)[:max_samples]
        self.data = []
        self.num_freq = num_freq
        self.num_random_points = num_random_points

        for sample in raw_data:
            # Normalize step
            step = (sample['steps'])

            # Handle the contour data which might be None
            contour = sample['processed_contours']
            fourier_descriptors = sample['feq']
            fourier_tensor = torch.tensor(fourier_descriptors, dtype=torch.complex64)

            contour_tensor = torch.tensor(contour, dtype=torch.float32)
            contour_tensor = contour_tensor.reshape(contour_tensor.shape[0]*contour_tensor.shape[1], contour_tensor.shape[2])
            contour_tensor = contour_tensor[:,:2]
            step = torch.tensor(step, dtype=torch.float32)
            circle_pts = even_circle_points(num_random_points,step)

            self.data.append(ContourDataPoint(
                step=step,
                fx=torch.tensor(sample['s1'], dtype=torch.float32).unsqueeze(0),
                processed_contour=contour_tensor,
                circle_points=circle_pts,
                feq=fourier_tensor,
                initial_domain =torch.tensor(sample['initial_domain'], dtype=torch.float32).unsqueeze(0),

            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def create_dataloaders(data_path, batch_size=32, shuffle=True, max_samples=800,num_random_points=500
                       ):

    full_dataset = ContourDataset(data_path, max_samples=max_samples,num_random_points=num_random_points)


    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader
# data_path = "F:\\Tumour\\picture\\boundary_data\\test_of _trans\\test_trans1.npy"
# train_loader = create_dataloaders(data_path, batch_size=300,
#                                                shuffle=True, max_samples=2000)
#
# # Example usage:
# data_path = "F:\\Tumour\\picture\\tumour_of _04_09_23_14_40\\interpolation_results.npy"
#
# # Option 1: Split by proportion (80% train, 20% test)
# train_loader, test_loader = create_dataloaders(data_path, batch_size=16,
#                                                shuffle=True, max_samples=800,
#                                                train_size=0.8)
#
# # Option 2: Split by exact numbers (600 train, 200 test)
# # train_loader, test_loader = create_dataloaders(data_path, batch_size=16,
# #                                               shuffle=True, max_samples=800,
# #                                               train_size=600)
#
# # Check the sizes
# print(f"Train batches: {len(train_loader)}")
# print(f"Test batches: {len(test_loader)}")
#
# # You can then iterate through the dataloaders like this:
# print("\nFirst training batch:")
# for batch in train_loader:
#     print("Batch steps:", batch.step.shape)
#     print("Batch s1 values:", batch.s1.shape)
#     print("Batch contour shapes:", batch.processed_contour.shape)
#     break
#
# print("\nFirst test batch:")
# for batch in train_loader:
#     print("Batch steps:", batch.step.shape)
#     print("Batch s1 values:", batch.s1.shape)
#     print("Batch contour shapes:", batch.processed_contour.shape)
#     print("Batch s2 values:", batch.feq.shape)
#     print("Batch input values:", batch.circle_points.shape)
#     break




# import matplotlib.pyplot as plt
#
# 1. Create the DataLoader (using your existing function)
