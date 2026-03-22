import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.Get_dataloader import create_data_loaders
from model.TP_model import TP_model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
from datetime import datetime
from datetime import timedelta
import time

l2_loss = LpLoss(size_average=False)

def train(epoches, lr, embed_dim = 256,seed = 42, log=None,save_dir=None,load_path=None):
    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = TP_model(num_layers=6, embed_dim=embed_dim)
    start_time = time.time()
    epoch_times = []
    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}.  Check model architecture or compatibility. Training from scratch.")
    m = m.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, amsgrad=True,weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

    if log is not None:
        writer = SummaryWriter(log_dir=log)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    global_step = 0

    for epoch in tqdm(range(epoches), desc="Training", unit="epoch"):
        epoch_start_time = time.time()
        train_T_loss = 0
        train_P_loss = 0
        for batch in train_loader:
            x_train = batch['x'].to(device)
            T_train = batch['T'].to(device)
            P_train = batch['P'].to(device)
            fx_train = batch['fx'].to(device)
            T_pred,P_pred = m(x_train,condition = fx_train)
            T_loss =l2_loss(T_pred, T_train)
            P_loss =l2_loss(P_pred, P_train)
            loss = T_loss + P_loss
            train_T_loss += T_loss.item()
            train_P_loss += P_loss.item()
            m.zero_grad()
            loss.backward()
            # clip_grad_norm_(m.parameters(), max_norm=1)
            optimizer.step()
            global_step += 1
            if log is not None:
                writer.add_scalar('Step/T', T_loss.item(), global_step)
                writer.add_scalar('Step/P', P_loss.item(), global_step)
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        test_T_loss = 0
        test_P_loss = 0
        with torch.no_grad():
           for batch in val_loader:
                phi_val = batch['x'].to(device)
                T_val = batch['T'].to(device)
                P_val = batch['P'].to(device)
                fx_val = batch['fx'].to(device)
                T_pred,P_pred = m(phi_val,condition = fx_val)
                T_loss = l2_loss(T_pred, T_val)
                P_loss = l2_loss(P_pred, P_val)
                test_T_loss += T_loss.item()
                test_P_loss += P_loss.item()
        # m.eval()

        if log is not None:
            writer.add_scalar('Train/T', train_T_loss/len(train_loader), epoch)
            writer.add_scalar('Train/P', train_P_loss/len(train_loader), epoch)
            writer.add_scalar('Test/T', test_T_loss/len(val_loader), epoch)
            writer.add_scalar('Test/P', test_P_loss/len(val_loader), epoch)
        if save_dir is not None and (epoch + 1) % 50 == 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"TPmodel.pth")
            torch.save(m.state_dict(), save_path)  # Save model state dictionary
            print(f"Model saved at {save_path}")
    total_time = time.time() - start_time

    if len(epoch_times) > 5:
        avg_epoch_time = sum(epoch_times[5:]) / len(epoch_times[5:])
    else:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

  
    if log is not None:
        writer.add_scalar('Time/total_training_time', total_time, 0)
        writer.add_scalar('Time/avg_epoch_time', avg_epoch_time, 0)
        writer.add_scalar('Time/min_epoch_time', min(epoch_times), 0)
        writer.add_scalar('Time/max_epoch_time', max(epoch_times), 0)
        writer.close()





if __name__ == '__main__':
    seed = 42
    output_file_path = r"testing_optimized_normalized.npz"
    uv_path = r"uv.npy"
    batchsize = 1000
    train_loader, val_loader = create_data_loaders(output_file_path, uv_path, batch_size=batchsize, random_seed=seed)
    print(f"train_loader: {len(train_loader)}")
    print(f"val_loader: {len(val_loader)}")
    print(f'the dataset size is {len(train_loader) * batchsize}')
    print('aoe_loss/len(val_loader)This representation precludes')

    save_dir = './Checkpoint/'
    base_log_dir = ('/root/tf-logs/12_6TP')
    load_path = ".\\pretrainmodel_posttrained.pth"
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "TP---" + time_based_unique_id)
    save_dir = os.path.join(save_dir, "TS2d_TP---" + time_based_unique_id)

    train(250, lr = 1e-3, embed_dim=512+128, log=log_dir, save_dir=save_dir, load_path=None)


    # plot(embed_dim=256, save_dir=save_dir, load_path=None)
