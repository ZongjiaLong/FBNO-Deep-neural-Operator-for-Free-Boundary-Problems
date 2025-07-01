import math
import optuna
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.Trans_data import create_dataloaders

from model.Pretrainmodel import Model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import matplotlib.pyplot as plt
from datetime import datetime
from torch.fft import fft, ifft
from torch.nn.utils import clip_grad_norm_



data_path = ".\\domain_train.npy"
test_path = ".\\domain_test.npy"
train_loader = create_dataloaders(data_path, batch_size=20,
                                               shuffle=False, max_samples=3000,num_random_points=300)
test_loader = create_dataloaders(test_path, batch_size=10,shuffle = False, max_samples=600,num_random_points=300)




# 示例用法
myloss = LpLoss(size_average=False)
MSE_loss = torch.nn.HuberLoss()
l2_loss = LpLoss(size_average=False)
l2_loss_train = LpLoss(size_average=False)


def plot_vertical_comparison(predicted, target, interior_points, mapped_interior, epoch, save_dir=None):
    """Plot predicted vs target boundaries with interior points visualization in vertical layout."""
    if save_dir is None:
        return

    # Create timestamped directory on first call
    if not hasattr(plot_vertical_comparison, "_initialized"):
        current_time = datetime.now().strftime("%m%d_%H%M%S")
        plot_vertical_comparison.time_based_dir = os.path.join(save_dir, f"plots---{current_time}")
        os.makedirs(plot_vertical_comparison.time_based_dir, exist_ok=True)
        plot_vertical_comparison._initialized = True

    # Convert tensors to numpy arrays
    if torch.is_tensor(predicted):
        predicted = predicted.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(interior_points):
        interior_points = interior_points.detach().cpu().numpy()
    if torch.is_tensor(mapped_interior):
        mapped_interior = mapped_interior.detach().cpu().numpy()

    # Plot each sample and timestep
    for sample_idx in range(predicted.shape[0]):
        sample_dir = os.path.join(plot_vertical_comparison.time_based_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        for timestep in range(predicted.shape[1]):
            # Create figure with transparent background and vertical subplots
            fig = plt.figure(figsize=(2, 6), facecolor='none')

            # Extract data for current sample and timestep
            pred = predicted[sample_idx, timestep]
            true = target[sample_idx, timestep]
            interior = interior_points[sample_idx, timestep]
            mapped = mapped_interior[sample_idx, timestep]

            # Plot 1 (Top): Original unit circle with interior points
            ax1 = plt.subplot(3, 1, 1)
            ax1.scatter(interior[:, 0], interior[:, 1], c='g', s=3, alpha=0.05)
            # Draw unit circle
            theta = np.linspace(0, 2 * np.pi, 100)
            ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
            ax1.axis('equal')
            ax1.set_facecolor('none')
            for spine in ax1.spines.values():
                spine.set_visible(False)
            # Remove axis ticks and labels
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Plot 2 (Middle): Output space with predicted boundary and mapped points
            ax2 = plt.subplot(3, 1, 2)
            ax2.scatter(mapped[:, 0], mapped[:, 1], c='g', s=3, alpha=0.05)
            # Plot predicted boundary
            ax2.plot(pred[:, 0], pred[:, 1], 'r-', linewidth=2, linestyle='--')
            ax2.axis('equal')
            ax2.set_facecolor('none')
            for spine in ax2.spines.values():
                spine.set_visible(False)
            # Remove axis ticks and labels
            ax2.set_xticks([])
            ax2.set_yticks([])

            # Plot 3 (Bottom): Output space with true boundary and mapped points
            ax3 = plt.subplot(3, 1, 3)
            ax3.scatter(mapped[:, 0], mapped[:, 1], c='g', s=3, alpha=0.05)
            # Plot true boundary
            ax3.plot(true[:, 0], true[:, 1], 'b-', linewidth=2)
            ax3.axis('equal')
            ax3.set_facecolor('none')
            for spine in ax3.spines.values():
                spine.set_visible(False)
            # Remove axis ticks and labels
            ax3.set_xticks([])
            ax3.set_yticks([])

            plt.tight_layout()

            # Save figure with transparent background
            plt.savefig(os.path.join(sample_dir, f"epoch_{epoch}_timestep_{timestep}.png"),
                        transparent=True, bbox_inches='tight', pad_inches=0,dpi = 100)
            plt.close()
def plot_comparison(predicted, target, interior_points, mapped_interior, epoch, save_dir=None):
    """Plot predicted vs target boundaries with interior points visualization."""
    if save_dir is None:
        return

    # Create timestamped directory on first call
    if not hasattr(plot_comparison, "_initialized"):
        current_time = datetime.now().strftime("%m%d_%H%M%S")
        plot_comparison.time_based_dir = os.path.join(save_dir, f"plots---{current_time}")
        os.makedirs(plot_comparison.time_based_dir, exist_ok=True)
        plot_comparison._initialized = True

    # Convert tensors to numpy arrays
    if torch.is_tensor(predicted):
        predicted = predicted.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(interior_points):
        interior_points = interior_points.detach().cpu().numpy()
    if torch.is_tensor(mapped_interior):
        mapped_interior = mapped_interior.detach().cpu().numpy()

    # Plot each sample and timestep
    for sample_idx in range(predicted.shape[0]):
        sample_dir = os.path.join(plot_comparison.time_based_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        for timestep in range(predicted.shape[1]):
            # Create figure with transparent background
            fig = plt.figure(figsize=(18, 6), facecolor='none')

            # Extract data for current sample and timestep
            pred = predicted[sample_idx, timestep]
            true = target[sample_idx, timestep]
            interior = interior_points[sample_idx, timestep]
            mapped = mapped_interior[sample_idx, timestep]

            # Plot 1: Original unit circle with interior points
            ax1 = plt.subplot(1, 3, 1)
            ax1.scatter(interior[:, 0], interior[:, 1], c='g', s=15, alpha=0.5)
            # Draw unit circle
            theta = np.linspace(0, 2 * np.pi, 100)
            ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
            # ax1.set_title('Input Space\nUnit Circle with Interior Points')
            ax1.axis('equal')
            # ax1.legend()
            # Set subplot background transparent
            ax1.set_facecolor('none')

            # Plot 2: Output space with mapped interior points
            ax2 = plt.subplot(1, 3, 2)
            ax2.scatter(mapped[:, 0], mapped[:, 1], c='g', s=15, alpha=0.5)
            # ax2.set_title('Output Space\nMapped Interior Points')
            ax2.axis('equal')
            # ax2.legend()
            # Set subplot background transparent
            ax2.set_facecolor('none')

            # Plot 3: Final comparison with boundaries
            ax3 = plt.subplot(1, 3, 3)
            # Plot mapped interior points (lighter color)
            ax3.scatter(mapped[:, 0], mapped[:, 1], c='g', s=15, alpha=0.2)
            # Plot predicted boundary
            # ax3.plot(pred[:, 0], pred[:, 1], 'r-', linewidth=2, linestyle='--')
            # Plot true boundary
            ax3.plot(true[:, 0], true[:, 1], 'b-', linewidth=2)
            # ax3.set_title('Final Comparison\nBoundaries with Interior Points')
            ax3.axis('equal')
            # ax3.legend()
            # Set subplot background transparent
            ax3.set_facecolor('none')

            plt.suptitle(f'Epoch {epoch}')
            plt.tight_layout()

            # Save figure with transparent background
            plt.savefig(os.path.join(sample_dir, f"epoch_{epoch}_timestep_{timestep}.svg"),
                       transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
def gradients(u, x, order=1):

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
def generate_interior_points(batch_size,t, device):
    r = torch.sqrt(torch.rand(15, batch_size, device=device))  # [10, batch_size]
    theta = 2 * math.pi * torch.rand(15, batch_size, device=device)  # [10, batch_size]
    x = r * torch.cos(theta)  # [10, batch_size]
    y = r * torch.sin(theta)  # [10, batch_size]
    t = t.squeeze().unsqueeze(1).expand(-1,  batch_size)
    points = torch.stack((x, y,t), dim=-1)  # [10, batch_size, 2]
    points = points.reshape(points.shape[0]*points.shape[1], -1)
    points = points.unsqueeze(0)
    return points
def gradient_loss(x_pred_interior,y_pred_interior,x_interior,y_interior):
    f1_interior = gradients(x_pred_interior, x_interior)
    f2_interior = gradients(x_pred_interior, y_interior)
    g1_interior = gradients(y_pred_interior, x_interior)
    g2_interior = gradients(y_pred_interior, y_interior)
    x_xx = gradients(f1_interior, x_interior)
    x_yy = gradients(f2_interior, y_interior)
    y_xx = gradients(g1_interior, x_interior)
    y_yy = gradients(g2_interior, y_interior)
    loss1_in = (f1_interior * g2_interior - f2_interior * g1_interior)
    loss1_1 = torch.mean(torch.relu(-loss1_in + 1e-4))
    loss1_sum = torch.sum(torch.relu(-loss1_in +0))
    laplacian = torch.mean(torch.abs(x_xx+x_yy)+torch.abs(y_xx+y_yy))
    # loss_gradient = laplacian*0.1+loss1_sum
    loss_gradient = laplacian*0.1+loss1_1+loss1_sum

    # loss_gradient = loss1_sum

    return loss_gradient
def f_loss(pred_contour, target_fd,num_freq=10):

    begin = target_fd[:, :, :num_freq]
    end = target_fd[:, :, -num_freq:]
    fft_coeffs = torch.cat((begin, end), dim=2)

    # num_freq = pred_fd.shape[2]
    complex_pred = pred_contour[:, :, :, 0] + 1j * pred_contour[:, :, :, 1]
    pred_fd = torch.fft.fft(complex_pred, dim=-1)  # No need to normalize, MSE handles scale.
    pred_begin = pred_fd[:, :, :num_freq]
    pred_end = pred_fd[:, :, -num_freq:]
    pred_coeffs = torch.cat((pred_begin, pred_end), dim=2)

    low_freq_loss_real = l2_loss(pred_coeffs.real, fft_coeffs.real)
    low_freq_loss_imag = l2_loss(pred_coeffs.imag, fft_coeffs.imag)

    return low_freq_loss_real,low_freq_loss_imag

def fourier_loss(pred_contour, target_fd, num_freq=64,name=  None):
    begin = target_fd[:, :, :num_freq]
    end = target_fd[:, :, -num_freq:]
    fft_coeffs = torch.cat((begin, end), dim=2)

    complex_pred = pred_contour[:, :,:, 0] + 1j * pred_contour[:, :,:, 1]
    pred_fd = torch.fft.fft(complex_pred,dim = -1)  # No need to normalize, MSE handles scale.
    pred_begin = pred_fd[:, :, :num_freq]
    pred_end = pred_fd[:, :, -num_freq:]
    pred_coeffs = torch.cat((pred_begin, pred_end), dim=2)

    if name is None:
        low_freq_loss_real = myloss(pred_coeffs.real,fft_coeffs.real)
        low_freq_loss_imag = myloss(pred_coeffs.imag, fft_coeffs.imag)
    else:
        low_freq_loss_real = l2_loss(pred_coeffs.real, fft_coeffs.real)
        low_freq_loss_imag = l2_loss(pred_coeffs.imag, fft_coeffs.imag)
    low_freq_loss = low_freq_loss_real + low_freq_loss_imag
    total_loss = low_freq_loss

    return total_loss

def train(epoches, lr, embed_dim = 64,seed = 420, log=None,save_dir=None,load_path=None):
    setup_seed(seed)
    m = Model(embed_dim = embed_dim,num_points=1000)
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
    steps_per_epoch = len(train_loader)
    print(steps_per_epoch)
    sfds = len(test_loader)
    print(sfds)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log is not None:
        writer = SummaryWriter(log_dir=log)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epoches), desc="Training", unit="epoch"):

        m = m.to('cuda')
        S_loss = 0
        total_loss = 0
        feq_loss = 0
        loss_gradient = 0
        decoder = 0
        for batch_idx, batch in enumerate(train_loader):
            t = batch.step.to(device)
            input_interior = generate_interior_points(200,t[:1], device)
            x_interior, y_interior,t_interior = input_interior[:, :, 0].requires_grad_(True), input_interior[:, :,1].requires_grad_(True),input_interior[:, :,2]
            target_data = batch.processed_contour.to(device)
            target_fd = batch.feq.to(device)
            ref_points = batch.circle_points.to(device)
            fx = batch.fx.to(device)
            initial_domain = batch.initial_domain.to(device)
            x_boundary, y_boundary,t_boundary = ref_points[:,:, 0], ref_points[:,:, 1], ref_points[:,:, 2]
            x_pred_boundary,y_pred_boundary,_,_ = m.forward(x = x_boundary, y = y_boundary,t= t_boundary,fx=fx,init_domain = initial_domain)
            pred_boundary = torch.stack([x_pred_boundary, y_pred_boundary], dim=2)
            freq_loss = fourier_loss(pred_boundary.reshape(-1,15,300,2), target_fd,num_freq =30,name=1)
            loss_sup = l2_loss(pred_boundary,target_data)
            x_pred_interior,y_pred_interior,x_de,y_de = m.forward(x = x_interior, y = y_interior,t= t_interior,fx=fx[:1],init_domain = initial_domain[:1])
            grad_loss = gradient_loss(x_pred_interior,y_pred_interior,x_interior,y_interior)
            decoder_loss = l2_loss(x_de,x_interior)+l2_loss(y_de,y_interior)
            loss =loss_sup+decoder_loss+grad_loss*10

            # loss =loss_sup*10+freq_loss+decoder_loss+grad_loss
            S_loss += loss_sup
            decoder += decoder_loss
            total_loss += loss.item()
            loss_gradient +=grad_loss
            feq_loss += freq_loss.item()
            m.zero_grad()
            loss.backward()
            # clip_grad_norm_(m.parameters(), max_norm=3)

            optimizer.step()
        scheduler.step()
        # m.eval()
        test_loss = 0
        MSE_loss_test = 0
        test_sup = 0
        with torch.no_grad():
            plot_samples = []
            plot_targets = []
            plot_interiors = []  # Store interior points
            plot_mapped_interiors = []
            for batch_idx, batch in enumerate(test_loader):
                target_boundary_test = batch.feq.to(device)
                test_points = batch.processed_contour.to(device)
                ref_points_test = batch.circle_points.to(device)
                t_test = batch.step.to(device)
                fx_test = batch.fx.to(device)
                init_domain_test = batch.initial_domain.to(device)
                x_boundary_test, y_boundary_test,t_boundary_test = ref_points_test[:,:, 0], ref_points_test[:, :,1],ref_points_test[:,:,2]
                x_pred_boundary_test, y_pred_boundary_test,_,_ = m.forward(x=x_boundary_test, y=y_boundary_test, t=t_boundary_test, fx=fx_test,init_domain=init_domain_test)
                pred_boundary_test = torch.stack([x_pred_boundary_test, y_pred_boundary_test], dim=2)
                u_loss_MSE,u_loss = f_loss(pred_boundary_test.reshape(-1,15,300,2), target_boundary_test)
                sup_test = l2_loss(pred_boundary_test,test_points)
                test_sup += sup_test.item()
                test_loss += u_loss.item()
                MSE_loss_test += u_loss_MSE.item()


                if batch_idx == 0 and (epoch+1 ) % 50 == 0:  # Only from first batch every 10 epochs
                    plot_points = 1000
                    interior_points = generate_interior_points(plot_points,t_test[:1], device)
                    interior_points = interior_points.repeat(t_test.shape[0], 1, 1)
                    x_interior, y_interior,t_interior = interior_points[:,  :, 0], interior_points[:, :, 1], interior_points[:, :, 2]
                    x_mapped, y_mapped,_,_ = m.forward(
                        x=x_interior, y=y_interior,
                        t=t_interior,
                        fx=fx_test,init_domain=init_domain_test
                    )
                    interior_points = interior_points[:,:,:2].reshape(-1,15,plot_points,2)
                    mapped_interior = torch.stack([x_mapped, y_mapped], dim=2)
                    mapped_interior = mapped_interior.reshape(-1,15,plot_points,2)
                    plot_samples.append(pred_boundary_test[:5].reshape(-1,15,300,2))
                    plot_targets.append(test_points[:5].reshape(-1,15,300,2))
                    plot_interiors.append(interior_points[:5])
                    plot_mapped_interiors.append(mapped_interior[:5])

                    # Plot comparison every 10 epochs
            if (epoch+1) % 50 == 0 and plot_samples:
                plot_comparison(
                    torch.cat(plot_samples, dim=0),
                    torch.cat(plot_targets, dim=0),
                    torch.cat(plot_interiors, dim=0),
                    torch.cat(plot_mapped_interiors, dim=0),
                    epoch + 1,
                    save_dir
                )

        train_loss_all = total_loss / len(train_loader)
        train_supervise = S_loss / len(train_loader)
        train_grad_all = loss_gradient / len(train_loader)
        train_frequency = feq_loss / len(train_loader)
        train_decoder = decoder / len(train_loader)
        sup = test_sup / len(test_loader)
        test_loss_all = test_loss / len(test_loader)
        train_loss_MSE_all = MSE_loss_test / len(test_loader)
        if log is not None:

            writer.add_scalar('Train/total', train_loss_all, epoch)
            writer.add_scalar('Train/supervise', train_supervise, epoch)
            writer.add_scalar('Train/Gradient', train_grad_all, epoch)
            writer.add_scalar('Train/frequency', train_frequency, epoch)
            writer.add_scalar('Train/decoder', train_decoder, epoch)

            writer.add_scalar('Test/sup', sup, epoch)
            writer.add_scalar("Test/Imag", test_loss_all, epoch)
            writer.add_scalar("Test/Real", train_loss_MSE_all, epoch)
        if save_dir is not None and (epoch + 1) % 50 == 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"pretrainmodel_posttrained.pth")
            torch.save(m.state_dict(), save_path)  # Save model state dictionary
            print(f"Model saved at {save_path}")
    if log is not None:
        writer.close()

def plot(embed_dim = 64,seed = 420,save_dir=None,load_path=None):
    setup_seed(seed)
    m = Model(embed_dim = embed_dim,num_points=1000)
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
    steps_per_epoch = len(train_loader)
    print(steps_per_epoch)
    sfds = len(test_loader)
    print(sfds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    MSE_loss_test = 0
    test_sup = 0
    with torch.no_grad():
        plot_samples = []
        plot_targets = []
        plot_interiors = []  # Store interior points
        plot_mapped_interiors = []
        for batch_idx, batch in enumerate(test_loader):
            target_boundary_test = batch.feq.to(device)
            test_points = batch.processed_contour.to(device)
            ref_points_test = batch.circle_points.to(device)
            t_test = batch.step.to(device)
            fx_test = batch.fx.to(device)
            init_domain_test = batch.initial_domain.to(device)
            x_boundary_test, y_boundary_test, t_boundary_test = ref_points_test[:, :, 0], ref_points_test[:, :,
                                                                                          1], ref_points_test[:, :, 2]
            x_pred_boundary_test, y_pred_boundary_test, _, _ = m.forward(x=x_boundary_test, y=y_boundary_test,
                                                                         t=t_boundary_test, fx=fx_test,
                                                                         init_domain=init_domain_test)
            pred_boundary_test = torch.stack([x_pred_boundary_test, y_pred_boundary_test], dim=2)
            u_loss_MSE, u_loss = f_loss(pred_boundary_test.reshape(-1, 15, 300, 2), target_boundary_test)
            sup_test = l2_loss(pred_boundary_test, test_points)
            test_sup += sup_test.item()
            test_loss += u_loss.item()
            MSE_loss_test += u_loss_MSE.item()

            if batch_idx == 0 :  # Only from first batch every 10 epochs
                plot_points = 1000
                interior_points = generate_interior_points(plot_points, t_test[:1], device)
                interior_points = interior_points.repeat(t_test.shape[0], 1, 1)
                x_interior, y_interior, t_interior = interior_points[:, :, 0], interior_points[:, :,
                                                                               1], interior_points[:, :, 2]
                x_mapped, y_mapped, _, _ = m.forward(
                    x=x_interior, y=y_interior,
                    t=t_interior,
                    fx=fx_test, init_domain=init_domain_test
                )
                interior_points = interior_points[:, :, :2].reshape(-1, 15, plot_points, 2)
                mapped_interior = torch.stack([x_mapped, y_mapped], dim=2)
                mapped_interior = mapped_interior.reshape(-1, 15, plot_points, 2)
                plot_samples.append(pred_boundary_test[:5].reshape(-1, 15, 300, 2))
                plot_targets.append(test_points[:5].reshape(-1, 15, 300, 2))
                plot_interiors.append(interior_points[:5])
                plot_mapped_interiors.append(mapped_interior[:5])

                # Plot comparison every 10 epochs
        if plot_samples:
            plot_vertical_comparison(
                torch.cat(plot_samples, dim=0),
                torch.cat(plot_targets, dim=0),
                torch.cat(plot_interiors, dim=0),
                torch.cat(plot_mapped_interiors, dim=0),
                1,
                save_dir
            )




if __name__ == '__main__':
    save_dir = '.\\save_dir'
    base_log_dir = ('.\\6_2')
    load_path = ".\\pretrainmodel_posttrained.pth"
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "fx---" + time_based_unique_id)
    # train(500, lr = 1e-5, embed_dim=256, log=log_dir, save_dir=save_dir, load_path=load_path)


    plot(embed_dim=256, save_dir=save_dir, load_path=load_path)
