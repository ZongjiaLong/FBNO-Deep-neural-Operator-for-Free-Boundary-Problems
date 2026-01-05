import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import random


class PhysicsDataset(Dataset):
    def __init__(self, data_samples,uv_file_path: str,is_training=True, sample_size=372):
        self.data = data_samples  # Now takes pre-loaded samples instead of file path
        self.uv_dict = np.load(uv_file_path, allow_pickle=True).item()
        self.uv_u = torch.tensor(self.uv_dict['x'], dtype=torch.float32)
        self.uv_v = torch.tensor(self.uv_dict['y'], dtype=torch.float32)
        self.is_training = is_training
        self.sample_size = sample_size
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        u_data = self.uv_u
        v_data = self.uv_v
        if self.is_training and self.sample_size > 0:

            T = torch.FloatTensor(sample['T'])
            x = torch.FloatTensor(sample['x'])
            y = torch.FloatTensor(sample['y'])
            P = torch.FloatTensor(sample['P'])
            params = torch.FloatTensor(sample['params'])
            indices = torch.randperm(x.shape[0])[:self.sample_size]
            x_sampled = x[indices]
            y_sampled = y[indices]
            u_sampled = u_data[indices]
            v_sampled = v_data[indices]
            T_sampled = T[indices]
            P_sampled = P[indices]
            return {
                'T': T_sampled,
                'phi': torch.cat([x_sampled. unsqueeze(1),y_sampled. unsqueeze(1)],dim=1),
                'x': torch.cat([u_sampled. unsqueeze(1),v_sampled. unsqueeze(1)],dim=1),
                'P': P_sampled,
                'fx': params,
            }
        else:
            T = torch.FloatTensor(sample['T'])
            x = torch.FloatTensor(sample['x'])
            y = torch.FloatTensor(sample['y'])
            P = torch.FloatTensor(sample['P'])
            params = torch.FloatTensor(sample['params'])
            return {
                'T': T,
                'phi': torch.cat([x. unsqueeze(1),y. unsqueeze(1)],dim=1), 'x': torch.cat([self.uv_u.unsqueeze(1),self.uv_v. unsqueeze(1)],dim=1),
                'P': P,
                'fx': params }


def create_data_loaders(data_file: str,uv_file_path: str, batch_size: int = 32,
                        train_ratio: float = 0.8, shuffle: bool = True,random_seed = 42, sample_size=200):
    """
    Create train and validation data loaders
    """
    all_data = np.load(data_file, allow_pickle=True)['samples']

    # Split the data indices
    dataset_size = len(all_data)
    indices = list(range(dataset_size))
    split = int(np.floor(train_ratio * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices = indices[:split]
    val_indices = indices[split:]

    # Create datasets with the split data
    train_data = [all_data[i] for i in train_indices]
    val_data = [all_data[i] for i in val_indices]

    train_dataset = PhysicsDataset(train_data, uv_file_path, is_training=True, sample_size=sample_size)
    val_dataset = PhysicsDataset(val_data, uv_file_path, is_training=False, sample_size=-1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
      # pin_memory=True,
      #   persistent_workers=True,
      #   prefetch_factor=1,
      #   num_workers=8,
      #   pin_memory_device='cuda'
                              )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False
        # ,pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=1,
        # num_workers=8,
        # pin_memory_device='cuda'
                            )

    return train_loader, val_loader
# Usage example:
if __name__ == "__main__":
    output_file_path = r"D:\desktop\output861\TS_data_normalized.npz"
    uv_path = r"D:\desktop\output861\TS_A_coord.npy"

    train_loader, val_loader = create_data_loaders(output_file_path,uv_path, batch_size=32)


    print(f"\n{'=' * 60}")
    print(f"OPTIMIZED EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"DataLoader ready - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Test one batch
    for sample_batch in val_loader:
        print(f"Batch shapes - P: {sample_batch['P'].shape}, "
              # f"T: {sample_batch['T'].shape}, "
              f"x: {torch.max(sample_batch['x']-sample_batch['phi'])} ")
              # f"phi: {sample_batch['phi'].shape}, "
              # f"fx: {sample_batch['fx'].shape}, ")
