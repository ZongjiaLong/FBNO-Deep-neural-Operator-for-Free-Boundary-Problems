import math
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from data.stefan_data import generate_data
from data.get_val_data import create_dataloaders
from model.stefan_model import Model
from utils.loss import setup_seed, count_params,LpLoss,gradients,batch_interp
from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
myloss = LpLoss(size_average=True)
MSE_loss = torch.nn.MSELoss()
l2_loss = LpLoss(size_average=True)

alpha = 1.15e-6
k_th = 2.18
rho = 917.0
L = 334000.0
k = k_th / (rho * L)

L_initial = 1.0
TotalTime = 3600.0
alpha = alpha * 2 * 15
K = 10000 * k
T_m = 0
T_l = 3
Nx = 101
Nt = 5001

L_max = 2.5
T_max = 20.0
Time_max = TotalTime
alpha_norm = alpha / (L_max ** 2) * Time_max
K_norm = K / (L_max ** 2) * Time_max * T_max
c_magnitude =  6e-4 * Time_max / L_max

def cool(t,period):
    return c_magnitude * (torch.sin(t * 0.00698/period * Time_max - math.pi/2)+1)
def Q_fx(x,q_magnitude):
    return q_magnitude * (torch.sin(3.14/100 * x*L_max) +1)
def Interior_PDE_loss(u,t,phi,x ,mag,s = None):
    phi_x = gradients(phi,x)
    loss = torch.mean(torch.relu(-phi_x+0.05))
    if s is not None:
        grad_penalty = torch.mean(torch.relu(torch.abs(phi_x) - 1.05*s))
        loss+=grad_penalty
    u_t  = gradients(u,t,order=1)
    u_phi_2 = gradients(u,phi,order=2)
    Q = Q_fx(phi,mag)
    pde1_left = u_t
    pde1_right = alpha_norm*u_phi_2+Q
    pde1loss = l2_loss(pde1_left,pde1_right)
    return loss,pde1loss

def Boundary_PDE_loss(u, t, phi,period):
    cool_cond = cool(t,period)
    phi_t = gradients(phi,t,order=1)
    u_phi = gradients(u,phi,order=1)
    pde2_left = -cool_cond
    pde2_right = phi_t+K_norm*u_phi
    pde2loss = l2_loss(pde2_left,pde2_right)
    return pde2loss


def train(epoches, lr, embed_dim = 64,seed = 3860, log=None,save_dir=None,load_path=None):
    setup_seed(seed)
    m = Model(embed_dim = embed_dim)
    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}.  Check model architecture or compatibility. Training from scratch.")
    m = m.to('cuda')
    a = count_params(m)
    print(a)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)    # steps_per_epoch = 2   # steps_per_epoch = 2
    if log is not None:
        writer = SummaryWriter(log_dir=log)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epoches), desc="Training", unit="epoch"):
        train_loader = generate_data(batchsize=30, n_s=350, n_interior=1800, n_boundary=400, n_initial=400)
        m = m.to('cuda')
        total_loss = 0
        interior_loss_accum = 0
        boundary_loss_accum = 0
        initial_loss_accum = 0
        boundary_right_loss_accum = 0
        interior_phi_accum = 0
        initial_phi_accum = 0
        boundary_left_loss_accum = 0
        for batch_idx, batch in enumerate(train_loader):
            interior = batch["interior"].to(device='cuda')
            left = batch["left"].to(device='cuda')
            right = batch["right"].to(device='cuda')
            initial = batch["initial"].to(device='cuda')
            mag = batch["mag"].to(device='cuda')
            period = batch["period"].to(device='cuda')
            interior_x_train = interior[:, :, 0].requires_grad_(True)
            interior_t_train = interior[:, :, 1].requires_grad_(True)
            left_x_train = left[:, :, 0].requires_grad_(True)
            right_x_train = right[:, :, 0].requires_grad_(True)
            left_t_train = left[:, :, 1].requires_grad_(True)
            right_t_train = right[:, :, 1].requires_grad_(True)
            initial_x_train = initial[:, :, 0].requires_grad_(True)
            initial_t_train = initial[:, :, 1].requires_grad_(True)
            t_1 = torch.zeros_like(right_x_train)

            phi_interiro,u_interior,tao_interior= m.forward(t = interior_t_train,mag=mag,period=period,x=interior_x_train)
            phi_left,u_left,_ = m.forward(t = left_t_train,mag=mag,period=period,x=left_x_train)
            phi_right,u_right,tao_right = m.forward(t = right_t_train,mag=mag,period=period,x=right_x_train)
            phi_initial,u_initial,_ = m.forward(t = initial_t_train,mag=mag,period=period,x=initial_x_train)
            phi_1,_,_ = m.forward(t = t_1,mag=mag,period=period,x=right_x_train)


            phi_loss,interior_loss = Interior_PDE_loss(u = u_interior,t = tao_interior,phi =phi_interiro,x = interior_x_train,mag = mag,s =phi_right )

            boundary_loss = Boundary_PDE_loss(u = u_right, t  = right_t_train, phi = phi_right,period=period)
            initial_cond = (T_l/T_max)*(1-phi_initial*L_max)**2
            initial_phi = myloss(phi_1,torch.ones_like(phi_1)/L_max)
            initial_loss = myloss(u_initial.squeeze(),initial_cond)
            boundary_right_loss = torch.mean(torch.abs(u_right))+torch.mean(torch.abs(phi_left))
            boundary_left_loss = myloss(u_left,torch.ones_like(u_left)*(T_l/T_max))

            losses = [interior_loss, boundary_loss, initial_loss, boundary_right_loss, boundary_left_loss,initial_phi,phi_loss]
            loss_names = ["interior_loss", "boundary_loss", 'initial_loss', 'boundary_right_loss', 'boundary_left_loss','initial_phi','phi_loss']

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
            # if phi_loss<1e-6:
            #     loss_weights['phi_loss']=0
            loss = loss_weights["interior_loss"] * interior_loss + loss_weights["boundary_loss"] * boundary_loss + \
                   loss_weights['initial_loss'] * initial_loss + loss_weights["boundary_right_loss"]  * boundary_right_loss + \
                   loss_weights["boundary_left_loss"] * boundary_left_loss+ loss_weights['phi_loss'] * phi_loss+loss_weights['initial_phi'] * initial_phi

            total_loss += loss.item()
            interior_phi_accum += phi_loss.item()
            interior_loss_accum += interior_loss.item()
            boundary_loss_accum += boundary_loss.item()
            initial_loss_accum += initial_loss.item()
            boundary_right_loss_accum += boundary_right_loss.item()
            boundary_left_loss_accum += boundary_left_loss.item()
            initial_phi_accum += initial_phi.item()
            m.zero_grad()
            loss.backward()
            # clip_grad_norm_(m.parameters(), max_norm=1)
            optimizer.step()
        scheduler.step()
        if epoch == epoches-1:
            u_test_loss = 0
            s_test_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    mag = batch.mag.to(device='cuda')
                    period = batch.period.to(device='cuda')
                    u_test = batch.u.to(device='cuda')
                    phi = batch.phi.to(device='cuda')
                    test_input = batch.test_input.to(device='cuda')
                    model_input = test_input.reshape(-1,test_input.shape[1]*test_input.shape[2],test_input.shape[3])
                    x_test = model_input[:, :, 0]
                    t_test = model_input[:, :, 1]

                    x_right = batch.x_right.to(device='cuda')
                    s_t_test = batch.t.to(device='cuda')
                    s_test = batch.s.to(device='cuda')

                    phi_out_test,u_test_hat,_ = m.forward(t = t_test,mag=mag,period=period,x = x_test)
                    phi_test = phi_out_test.reshape(u_test.shape)
                    u_act = batch_interp(queries=phi_test,xs = phi,ys = u_test)
                    u_test_hat = u_test_hat.reshape(u_act.shape)
                    u_loss = l2_loss(u_test_hat, u_act)


                    s_test_hat, _, _ = m.forward(t = s_t_test, mag=mag,period=period,x=x_right)
                    s_loss = l2_loss(s_test_hat, s_test)
                    u_test_loss += u_loss.item()
                    s_test_loss += s_loss.item()
                    u_test_loss_all = u_test_loss / len(test_loader)
                    s_test_loss_all = s_test_loss / len(test_loader)
                    print(f'the u loss is {u_test_loss_all}')
                    print(f'the s loss is {s_test_loss_all}')

        avg_total_loss = total_loss / len(train_loader)
        avg_interior_loss = interior_loss_accum / len(train_loader)
        avg_interior_phi_accum = interior_phi_accum / len(train_loader)
        avg_boundary_loss = boundary_loss_accum / len(train_loader)
        avg_initial_loss = initial_loss_accum / len(train_loader)
        avg_boundary_right_loss = boundary_right_loss_accum / len(train_loader)
        avg_boundary_left_loss = boundary_left_loss_accum / len(train_loader)
        avg_initial_phi_accum = initial_phi_accum / len(train_loader)


        # print(f"Epoch [{epoch + 1}/{epoches}], Training Loss: {avg_total_loss/len(train_loader)}")
        if log is not None:
            writer.add_scalar('Loss/Total', avg_total_loss, epoch)
            writer.add_scalar('Loss/Phi', avg_interior_phi_accum, epoch)
            writer.add_scalar('Loss/Initial_phi', avg_initial_phi_accum, epoch)
            writer.add_scalar('Loss/Interior', avg_interior_loss, epoch)
            writer.add_scalar('Loss/Boundary', avg_boundary_loss, epoch)
            writer.add_scalar('Loss/Initial', avg_initial_loss, epoch)
            writer.add_scalar('Loss/Boundary_Right', avg_boundary_right_loss, epoch)
            writer.add_scalar('Loss/Boundary_Left', avg_boundary_left_loss, epoch)
            # writer.add_scalar("Test/U", u_test_loss_all, epoch)
            # writer.add_scalar("Test/S", s_test_loss_all, epoch)
        if save_dir is not None and (epoch + 1) % 100 == 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"goodmodel.pth")
            torch.save(m.state_dict(), save_path)  # Save model state dictionary
            print(f"Model saved at {save_path}")


    mse_value1 = loss.item()
    return mse_value1


if __name__ == '__main__':
    test_loader = create_dataloaders(".\\stefan_data.npy", batch_size=1, shuffle=False,
                                     max_samples=300)

    save_dir = '.\\stefan600'
    base_log_dir = ('.\\base_log_dir')
    load_path = ".\\goodmodel.pth"
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "stefan---" + time_based_unique_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    mse_value1 = train(10000, lr = 1e-3, embed_dim=128, log=log_dir, save_dir=None, load_path=None)
