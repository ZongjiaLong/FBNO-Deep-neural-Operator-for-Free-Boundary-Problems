import numpy as np
import random
import torch
from torch.utils.data import DataLoader



def load_data(data,batch_size=32):
    class GeneratedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data["interior"])
        def __getitem__(self, idx):
            interior = self.data["interior"][idx]
            left = self.data["left"][idx]
            right = self.data["right"][idx]
            initial = self.data["initial"][idx]
            mag = self.data["mag"][idx]
            period = self.data["period"][idx]
            return {
                "interior": interior,
                'left': left,
                'right': right,
                'initial': initial,
                'mag': mag,
                'period': period
            }
    train_dataset = GeneratedDataset(data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

Time_max = 3600.0
T_max = 20.0
def generate_data(
        n_s=100,  # Default number of scenarios/cases
        n_interior=400,
        n_boundary = 200,
        n_initial = 100,
        batchsize=32):
    q_mag_min, q_mag_max = 0.0005, 0.0020
    q_period_min, q_period_max = 1.8, 6  # Range for q_period
    q_magnitude = torch.rand(n_s, 1) * (q_mag_max - q_mag_min) + q_mag_min

    q_magnitude = q_magnitude * Time_max / T_max
    q_period = torch.rand(n_s, 1) * (q_period_max - q_period_min) + q_period_min


    x_interior = torch.rand(n_s, n_interior, 1)  # Shape (n_s, n_interior, 1)
    # x_interior = x_interior**2
    t_interior = torch.rand(n_s, n_interior, 1)  # Shape (n_s, n_interior, 1)
    xt_interior = torch.cat((x_interior, t_interior), dim=2)

    x_left = torch.zeros(n_s, n_boundary, 1)  # x=0 for all
    t_left = torch.rand(n_s, n_boundary, 1)
    xt_left = torch.cat((x_left, t_left), dim=2)

    x_right = torch.ones(n_s, n_interior, 1)  # x=1 for all
    xt_right = torch.cat((x_right, t_interior), dim=2)

    x_initial = torch.rand(n_s, n_initial, 1)
    t_initial = torch.zeros(n_s, n_initial, 1)  # t=0 for all
    xt_initial = torch.cat((x_initial, t_initial), dim=2)
    data = {
        "interior": xt_interior,
        'left':xt_left,
        'right':xt_right,
        'initial':xt_initial,
        'mag': q_magnitude,
        'period': q_period,
    }

    train_loader = load_data(data, batch_size=batchsize)
    return train_loader
