import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.Get_dataloader import create_data_loaders
from model.H_model import TS_H_model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
from datetime import datetime
from datetime import timedelta
import time

l2_loss = LpLoss(size_average=False)
def train(epoches, lr, embed_dim = 256,seed = 42, log=None,save_dir=None,load_path=None):
    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = TS_H_model(num_layers=6, embed_dim=embed_dim)
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
        train_loss_phi = 0
        train_loss_aoe = 0
        for batch in train_loader:
            x_train = batch['x'].to(device)
            phi_train = batch['phi'].to(device)
            fx_train = batch['fx'].to(device)
            pred_phi,x_out = m(x = x_train,condition = fx_train)
            loss_phi =l2_loss(pred_phi, phi_train)
            aoe_loss = l2_loss(x_out, x_train)
            loss = loss_phi+aoe_loss
            train_loss_phi += loss_phi.item()
            train_loss_aoe += aoe_loss.item()
            m.zero_grad()
            loss.backward()
            # clip_grad_norm_(m.parameters(), max_norm=1)
            optimizer.step()
            global_step+=1
            # if log is not None:
            #     writer.add_scalar('Step/aoe', aoe_loss.item(), global_step)
            #     writer.add_scalar('Step/phi', loss_phi.item(), global_step)
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch} Loss {train_loss_phi / len(train_loader)} Time: {epoch_time:.2f}s')
        test_loss_phi = 0
        test_loss_aoe = 0
        with torch.no_grad():
           for batch in val_loader:
                x_val = batch['x'].to(device)
                phi_val = batch['phi'].to(device)
                fx_val = batch['fx'].to(device)
                pred_phi_val,x_out_val = m(x = x_val,condition = fx_val)
                loss_phi_val = l2_loss(pred_phi_val,phi_val)
                aoe_loss_val = l2_loss(x_out_val,x_val)
                test_loss_phi += loss_phi_val.item()
                test_loss_aoe += aoe_loss_val.item()
        # m.eval()

        if log is not None:
            writer.add_scalar('Train/phi', train_loss_phi/len(train_loader), epoch)
            writer.add_scalar('Train/aoe', train_loss_aoe/len(train_loader), epoch)
            writer.add_scalar('Val/phi', test_loss_phi/len(val_loader), epoch)
            writer.add_scalar('Val/aoe', test_loss_aoe/len(val_loader), epoch)
        if save_dir is not None and (epoch + 1) % 50 == 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"TSHmodel.pth")
            torch.save(m.state_dict(), save_path)  # Save model state dictionary
            print(f"Model saved at {save_path}")
    total_time = time.time() - start_time

    if len(epoch_times) > 5:
        avg_epoch_time = sum(epoch_times[5:]) / len(epoch_times[5:])
    else:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # 打印训练时间统计
    print("\n" + "=" * 50)
    print("训练时间统计:")
    print(f"总训练时间: {timedelta(seconds=int(total_time))}")
    print(f"总epoch数: {epoches}")
    print(f"平均每个epoch时间: {avg_epoch_time:.2f}秒")
    print(f"最快epoch时间: {min(epoch_times):.2f}秒")
    print(f"最慢epoch时间: {max(epoch_times):.2f}秒")
    print(f"总数据加载/处理时间估算: {timedelta(seconds=int(total_time - sum(epoch_times)))}")
    print("=" * 50)

    if log is not None:
        # 将时间统计也写入tensorboard
        writer.add_scalar('Time/total_training_time', total_time, 0)
        writer.add_scalar('Time/avg_epoch_time', avg_epoch_time, 0)
        writer.add_scalar('Time/min_epoch_time', min(epoch_times), 0)
        writer.add_scalar('Time/max_epoch_time', max(epoch_times), 0)
        writer.close()


if __name__ == '__main__':
    seed = 58
    output_file_path = r"D:\desktop\output861\TS_data_normalized.npz"
    uv_path = r"D:\desktop\output861\TS_A_coord.npy"
    batchsize = 100
    train_loader, val_loader = create_data_loaders(output_file_path, uv_path, batch_size=batchsize, random_seed=seed)
    print(f"train_loader: {len(train_loader)}")
    print(f"val_loader: {len(val_loader)}")
    print(f'the dataset size is {len(train_loader) * batchsize}')


    save_dir = 'D:\desktop\BCO\checkpoint\TS_2D'
    base_log_dir = ('D:\\desktop\\BCO\\tensorboard\\12_8')
    # load_path = './Checkpoint/TS2d---1206_080025/pretrainmodel.pth'
    load_path = "D:\download\TSHmodel.pth"
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "TS2d---" + time_based_unique_id)
    # save_dir = os.path.join(save_dir, "TS2d---" + time_based_unique_id)
    # train(150, lr = 1e-3, embed_dim=512+128, log=log_dir, save_dir=save_dir, load_path=None,seed=seed)
    train(2, lr=1e-8, embed_dim=512 + 128, log=None, save_dir=None, load_path=load_path, seed=seed)

    # plot(embed_dim=256, save_dir=save_dir, load_path=None)
