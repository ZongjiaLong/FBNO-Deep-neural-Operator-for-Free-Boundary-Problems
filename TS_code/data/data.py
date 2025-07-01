import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from collections import namedtuple

DataPoint = namedtuple('DataPoint', [
    'interior_points', 'interior_rou0', 'left_points', 'right_points',
    'left_T', 'right_T', 'left_rou0', 'right_rou0', 'initial_points', 'initial_T', 'initial_rou0','heat_para'
])

def load_data(data, batch_size=32):
    class GeneratedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data.interior_points)

        def __getitem__(self, idx):
            return DataPoint(
                interior_points=self.data.interior_points[idx],
                interior_rou0=self.data.interior_rou0[idx],
                left_points=self.data.left_points[idx],
                right_points=self.data.right_points[idx],
                left_T=self.data.left_T[idx],
                right_T=self.data.right_T[idx],
                left_rou0=self.data.left_rou0[idx],
                right_rou0=self.data.right_rou0[idx],

                initial_points=self.data.initial_points[idx],
                initial_T=self.data.initial_T[idx],
                initial_rou0=self.data.initial_rou0[idx],

                heat_para=self.data.heat_para[idx]
            )

    # Convert the dictionary to a named tuple
    data_tuple = DataPoint(
        interior_points=data["all_interior_points"],
        interior_rou0=data["all_interior_rou"],
        # interior_source=data['all_interior_source'],
        left_points=data['all_boundary_left'],
        right_points=data['all_boundary_right'],
        left_T=data['all_boundary_left_T'],
        right_T=data['all_boundary_right_T'],
        left_rou0=data['all_boundary_left_rou'],
        right_rou0=data['all_boundary_right_rou'],
        initial_points=data["all_initial_points"],
        initial_T=data["all_initial_T"],
        initial_rou0=data["all_initial_rou"],

        heat_para=data["all_heat_para"]
    )

    train_dataset = GeneratedDataset(data_tuple)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def func(x,A,B,C):

    return 15*(A / (1 + torch.exp(-B * x + B * C)))

def heat_source(x, t,A,B,C,D):
    a = D * torch.sin(A * x*(x+0.5*t)) * (torch.sin(C * (t+B)) +0.2)
    return a


def generate_data(
    n_s=1,  # Default number of scenarios/cases
    n_interior=400,  # Default number of interior points
    n_boundary=400,  # Default number of boundary points
    n_initial=500,  # Default number of initial points
    L=1.0,  # Default spatial domain size
    t_max=1.0,
    batchsize = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    expan_A_val = (0.89 + 0.96) / 2  # 0.925
    expan_B_val = (1.1 + 1.65) / 2 * 4  # 1.375
    expan_C_val = -(0.35 + 0.5) / 2  # -0.425
    expan_A = torch.full((n_s,1,), expan_A_val, device=device)
    expan_B = torch.full((n_s,1,), expan_B_val, device=device)
    expan_C = torch.full((n_s,1,), expan_C_val, device=device)

    heat_A = torch.rand(n_s,1, device=device) * (7 - 3) + 3
    heat_B = torch.rand(n_s,1, device=device) * (0.5 - 0) + 0
    heat_C = torch.rand(n_s,1, device=device) * (9 - 6.28) + 6.28
    heat_D = torch.full((n_s,1,), 4.5*12, device=device)

    all_heat_para = torch.stack([heat_A, heat_B, heat_C, heat_D], dim=1)

    x_interior = torch.rand(n_s, n_interior, device=device) * L
    t_interior = torch.rand(n_s, n_interior, device=device) * t_max
    xt_interior = torch.stack([x_interior, t_interior], dim=2)
    interior_T = torch.zeros_like(xt_interior)[:,:,:1].squeeze()
    interior_rou0 = func(x=interior_T, A=expan_A, B=expan_B,C=expan_C)



    t_boundary = torch.rand(n_s, n_boundary, device=device) * t_max
    x_boundary_left = torch.zeros(n_s,n_boundary, device=device)
    x_boundary_right = torch.ones(n_s,n_boundary, device=device) * L

    xt_boundary_left = torch.stack((x_boundary_left, t_boundary), dim=2)
    xt_boundary_right = torch.stack((x_boundary_right, t_boundary), dim=2)

    boundary_left_T = torch.zeros_like(xt_boundary_left)[:,:,:1].squeeze()
    boundary_right_T = torch.zeros_like(xt_boundary_right)[:,:,:1].squeeze()


    boundary_left_rou0 = func(x=boundary_left_T,A=expan_A,B=expan_B,C=expan_C)
    boundary_right_rou0 = func(x=boundary_right_T,A=expan_A,B=expan_B,C=expan_C)

    # 生成初始点 (t = 0)

    x_initial = torch.rand(n_s,n_initial, device=device) * L
    t_initial = torch.zeros(n_s,n_initial, device=device)

    xt_initial = torch.stack((x_initial, t_initial), dim=2)

    initial_T = torch.zeros_like(xt_initial)[:,:,:1].squeeze()
    initial_rou = func(x=initial_T, A=expan_A,B=expan_B,C=expan_C)




    data = {
        "all_interior_points": xt_interior,
        # "all_interior_source": interior_source,

        "all_initial_points": xt_initial,
        "all_initial_T": initial_T,

        "all_initial_rou": initial_rou,
        "all_interior_rou": interior_rou0,

        'all_boundary_left': xt_boundary_left,
        'all_boundary_right': xt_boundary_right,
        'all_boundary_left_T': boundary_left_T,
        'all_boundary_right_T': boundary_right_T,
        'all_boundary_left_rou': boundary_left_rou0,
        'all_boundary_right_rou': boundary_right_rou0,

        "all_heat_para":all_heat_para.squeeze(),

    }
    dataloader = load_data(data=data,batch_size =batchsize)

    return dataloader


