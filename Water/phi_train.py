import math

import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.phi.phi1_dataloaders import create_data_loaders
from model.phi_model import Model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

myloss = torch.nn.L1Loss(reduction='mean')
Huber_loss = torch.nn.HuberLoss()
l2_loss = LpLoss(size_average=False)



def train(epoches, lr, embed_dim = 64,seed = 420, log=None,save_dir=None,load_path=None):
    setup_seed(seed)
    m = Model(embed_dim = embed_dim,device = device)
    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}.  Check model architecture or compatibility. Training from scratch.")
    m = m.to('cuda')
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

    if log is not None:
        writer = SummaryWriter(log_dir=log)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epoches), desc="Training", unit="epoch"):
        m = m.to('cuda')
        S_loss_max = 0
        S_loss_zero = 0
        for batch_max,batch_zero in zip(max_train_loader,zero_train_loader):
            x_max = batch_max['input'].to(device)
            phi_max = batch_max['target'].to(device)
            t_max = batch_max['time'].to(device)
            fx_max = batch_max['fx'].to(device)
            phi_max_hat,_= m.fist_step(x = x_max,t=t_max,fx=fx_max)
            loss_max = myloss(phi_max_hat,phi_max)

            x_zero = batch_zero['input'].to(device)
            phi_zero = batch_zero['target'].to(device)
            t_zero = batch_zero['time'].to(device)
            fx_zero = batch_zero['fx'].to(device)
            _,phi_zero_hat = m.fist_step(x=x_zero, t=t_zero, fx=fx_zero)
            loss_zero = myloss(phi_zero_hat,phi_zero)
            loss = loss_max + loss_zero
            S_loss_max += loss_max
            S_loss_zero += loss_zero
            m.zero_grad()
            loss.backward()
            # clip_grad_norm_(m.parameters(), max_norm=3)

            optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch} Loss {S_loss_max/len(max_train_loader)}')
        S_loss_max_val = 0
        S_loss_zero_val = 0
        S_loss_max_val_H = 0
        S_loss_zero_val_H = 0

        with torch.no_grad():
            for batch_max_val, batch_zero_val in zip(max_val_loader, zero_val_loader):
                x_max_val = batch_max_val['input'].to(device)
                phi_max_val = batch_max_val['target'].to(device)
                t_max_val = batch_max_val['time'].to(device)
                fx_max_val = batch_max_val['fx'].to(device)
                phi_max_hat_val, _ = m.fist_step(x=x_max_val, t=t_max_val, fx=fx_max_val)
                loss_max_val = l2_loss(phi_max_hat_val, phi_max_val)
                loss_max_val_H = torch.mean(torch.abs(phi_max_hat_val - phi_max_val)) / torch.mean(torch.abs(phi_max_val))

                x_zero_val = batch_zero_val['input'].to(device)
                phi_zero_val = batch_zero_val['target'].to(device)
                t_zero_val = batch_zero_val['time'].to(device)
                fx_zero_val = batch_zero_val['fx'].to(device)
                _, phi_zero_hat_val = m.fist_step(x=x_zero_val, t=t_zero_val, fx=fx_zero_val)
                loss_zero_val = l2_loss(phi_zero_hat_val, phi_zero_val)
                loss_zero_val_H = torch.mean(torch.abs(phi_zero_hat_val-phi_zero_val))/torch.mean(torch.abs(phi_zero_val))

                S_loss_max_val += loss_max_val
                S_loss_zero_val += loss_zero_val
                S_loss_max_val_H += loss_max_val_H
                S_loss_zero_val_H += loss_zero_val_H
        train_max = S_loss_max / len(max_train_loader)
        train_zero = S_loss_zero / len(zero_train_loader)
        val_max = S_loss_max_val / len(max_val_loader)
        val_zero = S_loss_zero_val / len(zero_val_loader)
        val_max_H = S_loss_max_val_H / len(max_val_loader)
        val_zero_H = S_loss_zero_val_H / len(zero_val_loader)
        if log is not None:
            writer.add_scalar('Train/max', train_max, epoch)
            writer.add_scalar('Train/zero', train_zero, epoch)
            writer.add_scalar('Val/max', val_max, epoch)
            writer.add_scalar('Val/zero', val_zero, epoch)
            writer.add_scalar('Val/max_H', val_max_H, epoch)
            writer.add_scalar('Val/zero_H', val_zero_H, epoch)
        if save_dir is not None and (epoch + 1) % 100 == 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"phi_first_model.pth")
            torch.save(m.state_dict(), save_path)  # Save model state dictionary
            print(f"Model saved at {save_path}")
    if log is not None:
        writer.close()



if __name__ == '__main__':
    max_data_path ="\\iso_z_max.npz"
    zero_data_path =  "\\iso_z_zero.npz"
    seed = 3407
    max_train_loader, max_val_loader = create_data_loaders(max_data_path, batch_size=500,shuffle=True,seed = seed)
    zero_train_loader, zero_val_loader = create_data_loaders(max_data_path, batch_size=500,shuffle=True,seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    save_dir = '\water'
    base_log_dir = ('\\11_28')
    load_path = "\phi_first_model.pth"
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "fx---" + time_based_unique_id)

    train(500, lr = 1e-3, embed_dim=256, log=log_dir, save_dir=save_dir, load_path=None,seed=seed)
