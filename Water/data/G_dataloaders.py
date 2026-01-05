import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class FluidDataset(Dataset):
    def __init__(self, npz_file, train_ratio=0.8, train=True, seed=42,dtype = torch.float32):
        data = np.load(npz_file,allow_pickle=True)
        self.dtype = dtype
        structured_data = data['structured_data']
        np.random.seed(seed)
        torch.manual_seed(seed)
        indices = np.random.permutation(len(structured_data))
        split_idx = int(len(structured_data) * train_ratio)
        if train:
            self.data = structured_data[indices[:split_idx]]
        else:
            self.data = structured_data[indices[split_idx:]]

        print(f"{'Training' if train else 'Testing'} set size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        kw = torch.tensor(item['kw'], dtype=self.dtype)
        ko = torch.tensor(item['ko'], dtype=self.dtype)
        fx = torch.tensor([kw, ko], dtype=self.dtype)
        mask = torch.tensor(item['mask'], dtype=self.dtype)
        time = torch.tensor(item['time'], dtype=self.dtype)
        velocity = torch.tensor(item['v'], dtype=self.dtype)
        x = torch.tensor(item['coord'][:, 1], dtype=self.dtype)
        y = torch.tensor(item['coord'][:,0], dtype=self.dtype)
        z = torch.tensor(item['coord'][:, 2], dtype=self.dtype)
        return {
            'fx': fx,
            'time': time,
            'x': x,
            'y': y,
            'z': z,
            'mask': mask,
            'velocity': velocity
        }

def collate_fn(batch):
    fxs = torch.stack([item['fx'] for item in batch])
    times = torch.stack([item['time'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    x_list = torch.stack([item['x'] for item in batch])
    y_list = torch.stack([item['y'] for item in batch])
    z_list = torch.stack([item['z'] for item in batch])
    velocities = torch.stack([item['velocity'] for item in batch])
    return {
        'fx': fxs,
        'time': times.unsqueeze(-1),
        'x': x_list.unsqueeze(-1),
        'y': y_list.unsqueeze(-1),
        'z': z_list.unsqueeze(-1),
        'v': velocities,
        'mask': mask.unsqueeze(-1),
    }

def get_dataloaders(npz_file, batch_size=32, train_ratio=0.8, num_workers=4,seed = 42,dtype = torch.float32):
    train_dataset = FluidDataset(npz_file, train_ratio=train_ratio, train=True,seed = seed,dtype = dtype)
    test_dataset = FluidDataset(npz_file, train_ratio=train_ratio, train=False,seed = seed,dtype = dtype)
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=2 and num_workers > 0,
        prefetch_factor=True if num_workers > 0 else None,
        pin_memory_device='cuda'

    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=2 and num_workers > 0,
        prefetch_factor=True if num_workers > 0 else None,
        pin_memory_device='cuda'
    )

    return train_loader, test_loader
