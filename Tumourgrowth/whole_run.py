import math

import os
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.value_data import create_dataloaders
from model.Whole_model import Model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn.utils import clip_grad_norm_



data_path = ".\\domain_train.npy"
test_path = ".\\domain_test.npy"
train_loader = create_dataloaders(data_path, batch_size=15,
                                               shuffle=True, max_samples=1000)
test_loader = create_dataloaders(test_path, batch_size=10,shuffle = False, max_samples=200)

# 示例用法
myloss = LpLoss(size_average=False)
MSE_loss = torch.nn.HuberLoss()
l2_loss = LpLoss(size_average=False)
l2_loss_train = LpLoss(size_average=False)



def generate_interior_points(batch_size, device):
    # Generate random radii for all points (sqrt for uniform distribution in circle)
    r = torch.sqrt(torch.rand(15, batch_size, device=device))  # [10, batch_size]
    theta = 2 * math.pi * torch.rand(15, batch_size, device=device)  # [10, batch_size]
    x = r * torch.cos(theta)  # [10, batch_size]
    y = r * torch.sin(theta)  # [10, batch_size]

    points = torch.stack((x, y), dim=-1)  # [10, batch_size, 2]
    points = points.permute(1, 0, 2)  # [batch_size, 10, 2]
    points = points.unsqueeze(0)  # [1, batch_size, 10, 2]
    points = points.permute(0, 2, 1, 3)  # [1, 10, batch_size, 2]
    return points



def train(epoches, lr, embed_dim = 64,seed = 420, log=None,save_dir=None,load_path=None):
    setup_seed(seed)
    m = Model(embed_dim = embed_dim,pretrain_path ='.\\pretrainmodel_posttrained.pth')
    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}.  Check model architecture or compatibility. Training from scratch.")
    m = m.to('cuda')
    a = count_params_with_grad(m)
    print(a)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log is not None:
        writer = SummaryWriter(log_dir=log)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epoches), desc="Training", unit="epoch"):

        m = m.to('cuda')
        S_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            value = batch.value.to(device)
            coord = batch.coord.to(device)
            fx = batch.fx.to(device)
            initial_domain = batch.initial_domain.to(device)
            x,y,t = coord[:,:, 0], coord[:,:, 1], coord[:,:, 2]
            u_hat= m.forward(x = x,y=y,t=t,init_domain=initial_domain,fx=fx)
            loss_sup = l2_loss(value,u_hat)
            S_loss += loss_sup
            m.zero_grad()
            loss_sup.backward()
            clip_grad_norm_(m.parameters(), max_norm=3)

            optimizer.step()
        scheduler.step()
        test_sup = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                value_test = batch.value.to(device)
                coord_test = batch.coord.to(device)
                fx_test = batch.fx.to(device)
                initial_domain_test = batch.initial_domain.to(device)
                x_test, y_test, t_test = coord_test[:, :, 0], coord_test[:, :, 1], coord_test[:, :, 2]
                u_hat_test = m.forward(x = x_test,y=y_test,t=t_test,init_domain=initial_domain_test,fx=fx_test)
                loss_sup_test = l2_loss(u_hat_test,value_test)
                test_sup += loss_sup_test
        train_supervise = S_loss / len(train_loader)
        sup = test_sup / len(test_loader)
        if log is not None:
            writer.add_scalar('Train/supervise', train_supervise, epoch)
            writer.add_scalar('Test/sup', sup, epoch)
        if save_dir is not None and (epoch + 1) % 100 == 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"whole_model.pth")
            # torch.save(m.state_dict(), save_path)  # Save model state dictionary
            print(f"Model saved at {save_path}")
    if log is not None:
        writer.close()



if __name__ == '__main__':
    save_dir = '.\\save_dir'
    base_log_dir = ('.\\logdir')
    load_path = ".\\whole_model.pth"
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "fx---" + time_based_unique_id)
    # os.makedirs(log_dir, exist_ok=True)
    #just set log= None if you don't use tensorboard,but I highly recommend it
    train(1000, lr = 1e-3, embed_dim=256, log=None, save_dir=None, load_path=None)
