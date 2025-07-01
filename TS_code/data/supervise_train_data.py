import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import pickle

def load_supervise_data(batch_size=32,split_num = 0.2, data_path=None):

    data = np.load(data_path, allow_pickle=True).item()
    train_data = {key: value[:split_num] for key, value in data.items()}
    class GeneratedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data["all_interior_points"])

        def __getitem__(self, idx):
            interior_points = torch.tensor(self.data["all_interior_points"][idx]).clone().detach().to(torch.float32)
            interior_T = torch.tensor(self.data['all_interior_T'][idx]).clone().detach().to(torch.float32)
            all_interior_rou = torch.tensor(self.data['all_interior_rou'][idx]).clone().detach().to(torch.float32)
            all_interior_v = torch.tensor(self.data['all_interior_v'][idx]).clone().detach().to(torch.float32)
            all_interior_phi = torch.tensor(self.data['all_interior_phi'][idx]).clone().detach().to(torch.float32)
            heat_para = torch.tensor(self.data["all_heat_para"][idx]).clone().detach().to(torch.float32)


            return {
                "test_points": interior_points,
                'test_T': interior_T,
                'test_rho': all_interior_rou,
                'test_v': all_interior_v,
                'test_phi': all_interior_phi,
                'heat_para': heat_para,

            }

    train_dataset = GeneratedDataset(train_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 测试集不需要 shuffle

    return  train_loader
