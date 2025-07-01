import os
from datetime import datetime
import math
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from data.supervise_train_data import load_supervise_data
from data.data import generate_data
from data.supervise_train_data import load_supervise_data
from model.TS_model import Model
from utils.loss import count_params, LpLoss, count_params_with_grad,setup_seed
import itertools
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

test_data_path = ".\\test_200.npy"
test_loader= load_supervise_data(batch_size=50, data_path=test_data_path,split_num=200)


train_data_path = ".\\train_10.npy"
S_train_loader= load_supervise_data(batch_size=1, data_path=train_data_path,split_num=10)


myloss =torch.nn.MSELoss()
MSE_loss = torch.nn.MSELoss()
l2_loss = LpLoss(size_average=True)
l1=LpLoss(p=1,size_average=True)
def gradients(u, x, order=1):
    x = x
    u = u
    if order == 1:
        return torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
    else:
        return gradients(
            gradients(u, x), x, order=order - 1)

def compute_PDE_loss(rou0,u,t,T,rou,urou,x0 = None,phi = None,interior_source =None):

    rou_t  = gradients(rou,t,order=1)
    urou_x = gradients(urou,phi,order=1)
    # right_mass = (rou_t+urou_x)
    # zero_mass = torch.zeros_like(right_mass)
    loss_mass = l2_loss(rou_t,-urou_x)
    loss_mass = loss_mass

    #(rho[j,i+1]-rho[j,i])/dt
    #(rho[j+1,i]*v[j+1,i] - rho[j,i]*v[j,i])/phi[j+1,i]-phi[j,i]
    #(倒数第2）

    if interior_source is not None:
        lamda = 0.8 * 15
        T_t = gradients(T,t,order=1)
        T_x = gradients(T,phi,order=1)
        T_xx = gradients(T,phi,order=2)
        left = rou*(T_t+u*T_x)-lamda*T_xx
        right = interior_source
        # a = left/right-1
        loss_res = l1(left,right)

    else:
        loss_res = 0.0
    if x0 is not None:
        fai_x = gradients(phi,x0,order=1)
        right_fai = rou0/rou
        loss_fai  = l2_loss(fai_x,right_fai)
    else:
        loss_fai = 0.0


    penalty = torch.sum(torch.relu(-phi))
    loss_positive = penalty

    return loss_mass,loss_res,loss_fai,loss_positive


def heat_source(x, t,heat_para):
    A, B, C, D = torch.chunk(heat_para, chunks=4, dim=1)
    a = D * torch.sin(A * x*(x+0.5*t)) * (torch.sin(C * (t+B)) +0.2)
    return a.squeeze()


def train(epoches, lr, embed_dim = 64, log=None,load_path =None,seed = 389260,save_dir=None):
    setup_seed(seed)
    Heat_path = "D:\\desktop\\BCO\\checkpoint\\heat_model_new2\\heat_model.pth"
    m = Model(embed_dim=embed_dim, Heat_path=Heat_path)
    if load_path:  # Check if a load path is provided
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}.  Check model architecture or compatibility. Training from scratch.")
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

    a = count_params(m)
    print('whole parameters:', a)
    b = count_params_with_grad(m)
    print('learnable parameters:', b)
    c = (b/a)*100
    print('learnable parameters percent:', c)

    if log is not None:
        writer = SummaryWriter(log_dir=log)
    m = m.to('cuda')
    supervise_train_loader_cycled = itertools.cycle(S_train_loader)
    for epoch in tqdm(range(epoches), desc="Training", unit="epoch"):
        m.train(True)
        loss_mass_train = 0
        loss_res_train = 0
        loss_fai_train = 0
        loss_positive_train = 0
        loss_supervise_train = 0
        S_loss = 0

        train_loader = generate_data(n_boundary=400, n_initial=400,n_interior=2000,n_s=150, batchsize=15)
        for batch_idx, batch in enumerate(train_loader):
            supervise_batch = next(supervise_train_loader_cycled)
            train_points = supervise_batch["test_points"].to(device='cuda')
            train_phi = supervise_batch["test_phi"].to(device='cuda')
            train_T = supervise_batch["test_T"].to(device='cuda')
            train_v = supervise_batch["test_v"].to(device='cuda')

            heat_train = supervise_batch["heat_para"].to(device='cuda')

            x_train = train_points[:, :, 0]
            t_train = train_points[:, :, 1]

            phi_train_hat, T_train_hat, rho_train_hat, v_train_hat, urou, tao = m.forward(
                x=x_train, t=t_train,heat_para=heat_train)

            T_loss_l2 = l2_loss(T_train_hat, train_T)
            v_loss_l2 = l2_loss(v_train_hat, train_v)
            phi_loss_l2 = l2_loss(phi_train_hat, train_phi)
            loss_data = T_loss_l2 + v_loss_l2+phi_loss_l2

            interior_points = batch.interior_points.to(device='cuda')
            interior_rou0 = batch.interior_rou0.to(device='cuda')

            left_points=batch.left_points.to(device='cuda')
            right_points=batch.right_points.to(device='cuda')
            initial_points = batch.initial_points.to(device='cuda')


            heat_para = batch.heat_para.to(device='cuda')


            interior_x = interior_points[:, :, 0].requires_grad_(True)
            interior_t = interior_points[:, :, 1].requires_grad_(True)
            initial_x = initial_points[:, :, 0]
            initial_t = initial_points[:, :, 1]
            left_x = left_points[:, :, 0]
            left_t = left_points[:, :, 1]
            right_x = right_points[:, :, 0].requires_grad_(True)
            right_t = right_points[:, :, 1]


            phi_interior,T_interior,rou_interior,u_interior,urou_interior,tao_interior = m.forward(
                x=interior_x,  t=interior_t, heat_para = heat_para)
            interior_source = heat_source(x=phi_interior.unsqueeze(2),t= interior_t.unsqueeze(2), heat_para = heat_para.unsqueeze(2))

            loss_mass_interior,loss_res_interior,loss_phi_interior,loss_positive_interior= compute_PDE_loss(
                rou0 = interior_rou0,u = u_interior,t = tao_interior,T = T_interior,rou = rou_interior,x0 = interior_x,phi = phi_interior,
                urou = urou_interior,interior_source=interior_source)

            phi_boundary_left,T_boundary_left,rou_boundary_left,u_boundary_left,urou_boundary_left,tao_left = m.forward(
                x=left_x,  t=left_t,heat_para = heat_para)

            loss_b_left = torch.mean(torch.abs(T_boundary_left))
            loss_v = torch.mean(torch.abs(u_boundary_left))
            loss_phi_s = torch.mean(torch.abs(phi_boundary_left))

            phi_boundary_right, T_boundary_right, rou_boundary_right, u_boundary_right, urou_boundary_right,tao_right = m.forward(
                x=right_x, t=right_t,heat_para=heat_para)

            loss_b_right =torch.mean(torch.abs(T_boundary_right))


            phi_initial,T_initial,rou_initial,u_initial,urou_initial,tao_initial = m.forward(x=initial_x,t=initial_t,heat_para = heat_para)

            loss_i = torch.mean(torch.abs(T_initial))
            loss_i_phi = l2_loss(phi_initial,initial_x)
            loss_mass = loss_mass_interior
            loss_res = loss_res_interior
            loss_fai = loss_phi_interior
            loss_positive = loss_positive_interior+loss_i_phi

            loss_supervise = loss_b_left+loss_b_right + loss_i+loss_phi_s+loss_v
            loss_mass_train += loss_mass.item()
            loss_res_train += loss_res.item()
            loss_fai_train += loss_fai.item()
            loss_positive_train += loss_positive.item()
            loss_supervise_train += loss_supervise.item()

            losses = [loss_mass, loss_res,loss_positive,loss_supervise,loss_fai,loss_data]
            loss_names = ["mass", "res", 'positive','supervise','fai','loss_data']
            weights = {}
            loss_weights = {name: 1.0 for name in loss_names}
            moving_average = 0.8
            total_grad_norm = 0
            for loss, name in zip(losses, loss_names):
                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    grad = torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) for p in
                                      m.parameters()])
                    total_grad_norm += torch.norm(grad, p=1)
                    weights[name] = grad.norm(p=1)

            for name in loss_names:
                if weights.get(name) is not None and weights[name] > 0:
                    weight = (total_grad_norm / weights[name])
                    loss_weights[name] = moving_average * weight + (1 - moving_average) * loss_weights[name]

            loss = loss_weights["mass"] * loss_mass + loss_weights["res"] * loss_res  +\
               loss_weights['positive'] * loss_positive + loss_weights["supervise"]* loss_supervise + \
               loss_weights["fai"] * loss_fai+loss_weights['loss_data']*loss_data
            # loss = loss_supervise

            m.zero_grad()
            loss.backward()
            # clip_grad_norm_(m.parameters(), max_norm=1,norm_type=2)
            optimizer.step()
            S_loss += loss.item()
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
                test_T = batch["test_T"].to(device='cuda')
                test_rho = batch["test_rho"].to(device='cuda')
                test_v = batch["test_v"].to(device='cuda')
                test_phi = batch["test_phi"].to(device='cuda')
                heat_test=batch["heat_para"].to(device='cuda')

                x_test = test_points[:, :, 0]
                t_test = test_points[:, :, 1]

                phi_hat, T_hat, rho_hat, v_hat, urou,tao = m.forward(
                    x=x_test, t=t_test,heat_para=heat_test)

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
                #
                # if epoch%100==0 and log is not None:
                #     mean_test_phi_squared = torch.mean(test_phi ** 2)
                #     mean_test_v_squared = torch.mean(test_v ** 2)
                #     mean_test_rho_squared = torch.mean(test_rho ** 2)
                #     mean_test_T_squared = torch.mean(test_T ** 2)
                #
                #     phi_diff = ((phi_hat - test_phi) ** 2/mean_test_phi_squared).detach().cpu().numpy().flatten()
                #     v_diff = ((v_hat - test_v) ** 2/mean_test_v_squared).detach().cpu().numpy().flatten()
                #     rho_diff = ((rho_hat - test_rho) ** 2/mean_test_rho_squared).detach().cpu().numpy().flatten()
                #     T_diff = ((T_hat - test_T) ** 2/mean_test_T_squared).detach().cpu().numpy().flatten()
                #
                #     plt.figure(figsize=(16, 12))
                #
                #     plt.subplot(2, 2, 1)
                #     plt.scatter(t_test.detach().cpu().numpy().flatten(), x_test.detach().cpu().numpy().flatten(),
                #                 c=phi_diff, cmap='viridis', s=10)
                #     plt.colorbar(label="Square Difference")
                #     plt.title(f"Epoch {epoch}, Batch {batch_idx}| Phi Square Difference")
                #     plt.xlabel("t")
                #     plt.ylabel("x")
                #
                #     plt.subplot(2, 2, 2)
                #     plt.scatter(t_test.detach().cpu().numpy().flatten(), x_test.detach().cpu().numpy().flatten(),
                #                 c=v_diff, cmap='viridis', s=10)
                #     plt.colorbar(label="Square Difference")
                #     plt.title(f"Epoch {epoch}, Batch {batch_idx} | V Square Difference")
                #     plt.xlabel("t")
                #     plt.ylabel("x")
                #
                #     plt.subplot(2, 2, 3)
                #     plt.scatter(t_test.detach().cpu().numpy().flatten(), x_test.detach().cpu().numpy().flatten(),
                #                 c=rho_diff, cmap='viridis', s=10)
                #     plt.colorbar(label="Square Difference")
                #     plt.title(f"Epoch {epoch}, Batch {batch_idx} | Rho Square Difference")
                #     plt.xlabel("t")
                #     plt.ylabel("x")
                #
                #     plt.subplot(2, 2, 4)
                #     plt.scatter(t_test.detach().cpu().numpy().flatten(), x_test.detach().cpu().numpy().flatten(),
                #                 c=T_diff, cmap='viridis', s=10)
                #     plt.colorbar(label="Square Difference")
                #     plt.title(f"Epoch {epoch}, Batch {batch_idx} | T Square Difference")
                #     plt.xlabel("t")
                #     plt.ylabel("x")
                #
                #     plt.tight_layout()
                #     writer.add_figure(f'Error Distribution/ Batch {batch_idx}', plt.gcf(), global_step=epoch)
                #     plt.close()

        mass_train = loss_mass_train / len(train_loader)
        res_train = loss_res_train / len(train_loader)
        fai_train = loss_fai_train / len(train_loader)
        positive_train = loss_positive_train / len(train_loader)
        supervise_train = loss_supervise_train / len(train_loader)

        phi_l2 = phi_l2 / len(test_loader)
        phi_mse = phi_mse / len(test_loader)
        v_l2 = v_l2 / len(test_loader)
        v_mse = v_mse / len(test_loader)
        rho_l2 = rho_l2 / len(test_loader)
        rho_mse = rho_mse / len(test_loader)
        T_l2 = T_l2 / len(test_loader)
        T_mse = T_mse / len(test_loader)
        if save_dir is not None and (epoch + 1) % 100 == 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"PI_model.pth")
            torch.save(m.state_dict(), save_path)  # Save model state dictionary
            print(f"Model saved at {save_path}")


        if log is not None:
            # mean_test_phi_squared = torch.mean(test_phi  )
            # mean_test_v_squared = torch.mean(test_v  )
            # mean_test_T_squared = torch.mean(test_T  )

            # phi_diff = ((phi_hat - test_phi)   / mean_test_phi_squared).detach().cpu().numpy().flatten()
            # v_diff = ((v_hat - test_v)   / mean_test_v_squared).detach().cpu().numpy().flatten()
            # T_diff = ((T_hat - test_T)   / mean_test_T_squared).detach().cpu().numpy().flatten()
            # writer.add_histogram('Distribution of error/phi', phi_diff, epoch)
            # writer.add_histogram('Distribution of error/v', v_diff, epoch)
            # writer.add_histogram('Distribution of error/T',T_diff, epoch)
            writer.add_scalar('Train/all', S_loss/len(train_loader), epoch)
            writer.add_scalar('Train/mass', mass_train, epoch)
            writer.add_scalar('Train/res', res_train, epoch)
            writer.add_scalar('Train/fai', fai_train, epoch)
            writer.add_scalar('Train/positive', positive_train, epoch)
            writer.add_scalar('Train/supervised', supervise_train, epoch)

            writer.add_scalar("Test_l2/phi", phi_l2, epoch)
            writer.add_scalar("Test_l2/v", v_l2, epoch)
            writer.add_scalar("Test_l2/rho", rho_l2, epoch)
            writer.add_scalar("Test_l2/T", T_l2, epoch)

            writer.add_scalar("Test_mse/phi", phi_mse, epoch)
            writer.add_scalar("Test_mse/v", v_mse, epoch)
            writer.add_scalar("Test_mse/rho", rho_mse, epoch)
            writer.add_scalar("Test_mse/T", T_mse, epoch)
            writer.add_scalar('Test_mse/Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            # for name, weight in loss_weights.items():
            #     writer.add_scalar(f"Loss_Weights/{name}", weight, epoch)


    if log is not None:
        writer.close()

    mse_value1 = loss.item()
    return mse_value1



if __name__ == '__main__':
    save_dir = '.\\save_dir'
    load_path = ".\\PI_model.pth "
    base_log_dir = '.\\base_log_dir'
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "PI---" + time_based_unique_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    mse_value1 = train(10000, lr = 1e-4, embed_dim=256,save_dir=save_dir, log=log_dir,load_path=None)







