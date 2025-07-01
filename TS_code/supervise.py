from datetime import datetime
import os

import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from data.supervise_train_data import load_supervise_data
from model.TS_model import Model
from utils.loss import setup_seed, count_params, LpLoss, count_params_with_grad
import matplotlib.pyplot as plt

train_data_path = ".\\train_10.npy"
test_data_path = ".\\test_200.npy"

train_loader= load_supervise_data(batch_size=1, data_path=train_data_path,split_num=10)
test_loader= load_supervise_data(batch_size=200, data_path=test_data_path,split_num=200)
myloss = torch.nn.MSELoss()
MSE_loss = torch.nn.MSELoss()
l2_loss = LpLoss(size_average=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(epoches, lr, embed_dim = 64,seed = 389260, log=None,save_dir = None):
    setup_seed(seed)
    m = Model(embed_dim = embed_dim)
    a = count_params(m)
    print('whole parameters:', a)
    b = count_params_with_grad(m)
    print('learnable parameters:', b)
    c = (b/a)*100
    print('learnable parameters percent:', c)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

    if log is not None:
        writer = SummaryWriter(log_dir=log)
    m = m.to('cuda')
    for epoch in tqdm(range(epoches), desc="Training", unit="epoch"):
        m.train(True)
        S_loss = 0
        phi_train_loss_sum = 0
        T_train_loss_sum = 0
        v_train_loss_sum = 0
        for batch_idx, batch in enumerate(train_loader):
            train_points = batch["test_points"].to(device='cuda')
            train_phi = batch["test_phi"].to(device='cuda')
            train_T = batch["test_T"].to(device='cuda')
            train_v = batch["test_v"].to(device='cuda')
            heat_train = batch["heat_para"].to(device='cuda')
            x_train = train_points[:, :, 0]
            t_train = train_points[:, :, 1]
            phi_train_hat, T_train_hat, rho_train_hat, v_train_hat, urou ,tao= m.forward(x=x_train, t=t_train, heat_para=heat_train)

            phi_loss_l2 = l2_loss(phi_train_hat, train_phi)
            T_loss_l2 = l2_loss(T_train_hat, train_T)
            v_loss_l2 = l2_loss(v_train_hat, train_v)
            loss = phi_loss_l2+T_loss_l2+v_loss_l2

            m.zero_grad()
            loss.backward()
            # clip_grad_norm_(m.parameters(), max_norm=0.8)
            optimizer.step()
            S_loss += loss.item()
            phi_train_loss_sum += phi_loss_l2.item()
            T_train_loss_sum += T_loss_l2.item()
            v_train_loss_sum += v_loss_l2.item()
        scheduler.step()
        m.eval()
        phi_l2 = 0
        phi_mse = 0
        v_l2 = 0
        v_mse = 0
        T_l2 = 0
        T_mse = 0
        rho_l2 = 0
        rho_mse = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                test_points = batch["test_points"].to(device='cuda')
                test_T=batch["test_T"].to(device='cuda')
                test_rho=batch["test_rho"].to(device='cuda')
                test_v=batch["test_v"].to(device='cuda')
                test_phi=batch["test_phi"].to(device='cuda')
                heat_test=batch["heat_para"].to(device='cuda')
                x_test = test_points[:, :, 0]
                t_test = test_points[:, :, 1]
                phi_hat, T_hat, rho_hat, v_hat, urou ,tao= m.forward(
                    x=x_test, t=t_test, heat_para=heat_test)
                phi_loss_l2 = l2_loss(phi_hat, test_phi)
                phi_loss_mse = MSE_loss(phi_hat, test_phi)
                v_loss_l2 = l2_loss(v_hat, test_v)
                v_loss_mse = MSE_loss(v_hat, test_v)
                rho_loss_l2 = l2_loss(rho_hat, test_rho)
                rho_loss_mse = MSE_loss(rho_hat, test_rho)
                T_loss_l2 = l2_loss(T_hat, test_T)
                T_loss_mse = MSE_loss(T_hat, test_T)
                phi_l2+=phi_loss_l2
                phi_mse+=phi_loss_mse
                v_l2+=v_loss_l2
                v_mse+=v_loss_mse
                T_l2+=T_loss_l2
                T_mse+=T_loss_mse
                rho_l2+=rho_loss_l2
                rho_mse+=rho_loss_mse
                # phi_error = (phi_hat - test_phi).cpu().numpy().flatten()
                # v_error = (v_hat - test_v).cpu().numpy().flatten()
                # rho_error = (rho_hat - test_rho).cpu().numpy().flatten()
                # T_error = (T_hat - test_T).cpu().numpy().flatten()

                # Log histograms to TensorBoard
                # writer.add_histogram('Distribution/phi_error', phi_error, batch_idx)
                # writer.add_histogram('Distribution/v_error', v_error, batch_idx)
                # writer.add_histogram('Distribution/rho_error', rho_error, batch_idx)
                # writer.add_histogram('Distribution/T_error', T_error, batch_idx)

                # if log is not None and batch_idx<=2:
                #     writer.add_histogram(f'Distribution of phi/error_batch{batch_idx}', (phi_hat - test_phi).detach().cpu().numpy(), epoch)
                #     writer.add_histogram(f'Distribution of v/error_batch{batch_idx}', (v_hat - test_v).detach().cpu().numpy(), epoch)
                #     writer.add_histogram(f'Distribution of T/error_batch{batch_idx}', (T_hat - test_T).detach().cpu().numpy(), epoch)


        phi_l2 = phi_l2 / len(test_loader)
        phi_mse = phi_mse / len(test_loader)
        v_l2 = v_l2 / len(test_loader)
        v_mse = v_mse / len(test_loader)
        rho_l2 = rho_l2 / len(test_loader)
        rho_mse = rho_mse / len(test_loader)
        T_l2 = T_l2 / len(test_loader)
        T_mse = T_mse / len(test_loader)


        if log is not None:
            # mean_test_phi_squared = torch.mean(test_phi)
            # mean_test_v_squared = torch.mean(test_v)
            # mean_test_T_squared = torch.mean(test_T)
            #
            # phi_diff = ((phi_hat - test_phi) / mean_test_phi_squared).detach().cpu().numpy().flatten()
            # v_diff = ((v_hat - test_v) / mean_test_v_squared).detach().cpu().numpy().flatten()
            # T_diff = ((T_hat - test_T) / mean_test_T_squared).detach().cpu().numpy().flatten()
            # writer.add_histogram('Distribution of error/phi',
            #                      phi_diff, epoch)
            # writer.add_histogram('Distribution of error/v',
            #                      v_diff, epoch)
            # writer.add_histogram('Distribution of error/T',
            #                      T_diff, epoch)
            writer.add_scalar('Train/l2', S_loss/len(train_loader), epoch)
            writer.add_scalar('Train/phi_loss', phi_train_loss_sum / len(train_loader), epoch)
            writer.add_scalar('Train/T_loss', T_train_loss_sum / len(train_loader), epoch)
            writer.add_scalar('Train/v_loss', v_train_loss_sum / len(train_loader), epoch)

            writer.add_scalar("Test_l2/phi", phi_l2, epoch)
            writer.add_scalar("Test_l2/v", v_l2, epoch)
            writer.add_scalar("Test_l2/rho", rho_l2, epoch)
            writer.add_scalar("Test_l2/T", T_l2, epoch)

            writer.add_scalar("Test_mse/phi", phi_mse, epoch)
            writer.add_scalar("Test_mse/v", v_mse, epoch)
            writer.add_scalar("Test_mse/rho", rho_mse, epoch)
            writer.add_scalar("Test_mse/T", T_mse, epoch)
        if save_dir is not None and (epoch + 1) % 100== 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"Supervise.pth")
            torch.save(m.state_dict(), save_path)  # Save model state dictionary
            print(f"Model saved at {save_path}")

    if log is not None:
        writer.close()

    mse_value1 = loss.item()
    return mse_value1



if __name__ == '__main__':
    save_dir = '.\\new_data'
    base_log_dir = '.\\5_271'
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "supervise---" + time_based_unique_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    mse_value1 = train(10000, lr = 1e-4, embed_dim=256,save_dir=save_dir, log=log_dir)
