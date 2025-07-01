import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from data.stefan_data import generate_data
from data.get_val_data import create_dataloaders
from model.stefan_model import Model
from utils.loss import setup_seed, count_params,LpLoss,gradients
from scipy.interpolate import griddata
import os
import seaborn as sns
l2_loss = LpLoss(size_average=False)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

total_time= 3600.0
def batch_interp(queries, xs, ys):
    """
    Vectorized interpolation for batchsize x t times.

    Args:
        queries: Tensor of shape (batchsize, t, n_queries) - query points for each batch and time
        xs: Tensor of shape (batchsize, t, n_points) - x coordinates for each batch and time
        ys: Tensor of shape (batchsize, t, n_points) - y coordinates for each batch and time

    Returns:
        Tensor of shape (batchsize, t, n_queries) - interpolated values
    """
    queries = queries.transpose(1, 2)
    xs = xs.transpose(1, 2)
    ys = ys.transpose(1, 2)

    batchsize, t, n_points = xs.shape
    _, _, n_queries = queries.shape

    # Sort each batch of x and corresponding y
    sorted_xs, indices = torch.sort(xs, dim=2)
    sorted_ys = torch.gather(ys, 2, indices)

    # Find where query values would fit in x (shape: (batchsize, t, n_queries))
    # idx_right = torch.searchsorted(sorted_xs, queries)
    idx_right = torch.searchsorted(sorted_xs.contiguous(), queries.contiguous())
    idx_left = idx_right - 1

    # Clamp indices to avoid out-of-bounds
    idx_left = torch.clamp(idx_left, 0, n_points - 1)
    idx_right = torch.clamp(idx_right, 0, n_points - 1)

    # Create index tensors for gather
    batch_indices = torch.arange(batchsize, device=xs.device)[:, None, None].expand(batchsize, t, n_queries)
    t_indices = torch.arange(t, device=xs.device)[None, :, None].expand(batchsize, t, n_queries)

    # Gather left/right x and y values
    x_left = sorted_xs[batch_indices, t_indices, idx_left]
    x_right = sorted_xs[batch_indices, t_indices, idx_right]
    y_left = sorted_ys[batch_indices, t_indices, idx_left]
    y_right = sorted_ys[batch_indices, t_indices, idx_right]

    # Avoid division by zero (if x_left == x_right)
    alpha = torch.zeros_like(queries)
    mask = x_left != x_right
    alpha[mask] = (queries[mask] - x_left[mask]) / (x_right[mask] - x_left[mask])

    # Linear interpolation
    return (y_left + alpha * (y_right - y_left)).transpose(1,2)

def compute_metric(left_top,right_top,right_bottom):
    left_top -=1
    right_top -=1
    right_bottom -=1
    ans = left_top**2 + 2*right_top**2 +right_bottom**2
    return (torch.sum(ans))**0.5


def vis_boundary_loss(test_loader, model_path,lablesize = 8,size=(2.65,2)):
    m = Model(embed_dim=128)
    m.load_state_dict(torch.load(model_path))
    m = m.to('cuda')
    m.eval()
    different_values = []
    u_loss_values = []
    s_loss_values = []

    # Remove torch.no_grad() since we need gradients for phi_x computation
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            mag = batch.mag.to(device='cuda')
            period = batch.period.to(device='cuda')
            u_test = batch.u.to(device='cuda')
            phi = batch.phi.to(device='cuda')
            test_input = batch.test_input.to(device='cuda')
            test_input = test_input[:,::]
            x_compute = test_input[:, :, :, 0]
            model_input = test_input.reshape(-1, test_input.shape[1] * test_input.shape[2], test_input.shape[3])

            # Ensure x_test has requires_grad=True
            x_test = model_input[:, :, 0].clone().requires_grad_(True)
            t_test = model_input[:, :, 1]

            x_right = batch.x_right.to(device='cuda')
            s_t_test = batch.t.to(device='cuda')
            s_test = batch.s.to(device='cuda')

            s_prime = torch.diff(s_test) / torch.diff(s_t_test)
            pad = s_prime[:, -1]
            s_prime = torch.cat([s_prime, pad.unsqueeze(0)], dim=1)
            s_prime = s_prime.unsqueeze(1).expand(-1, 101, -1)
            s_compute = s_test.unsqueeze(1).expand(-1, 101, -1)
            left_top = s_compute ** 2
            right_top = x_compute * s_compute * s_prime
            right_bottom = (x_compute ** 2) * (s_prime ** 2)
            dis = compute_metric(left_top, right_top, right_bottom)

            # Forward pass
            phi_out_test, u_test_hat, _ = m.forward(t=t_test, mag=mag, period=period, x=x_test)


            # Rest of the computation can be done without gradients
            phi_test = phi_out_test.reshape(u_test.shape)
            u_act = batch_interp(queries=phi_test, xs=phi, ys=u_test)
            u_test_hat = u_test_hat.reshape(u_act.shape)
            u_act *= 20
            u_test_hat *= 20
            u_loss = l2_loss(u_test_hat, u_act)
            s_test_hat, _, _ = m.forward(t=s_t_test, mag=mag, period=period, x=x_right)

            x_hat = phi_out_test.reshape(u_test.shape)
            x_hat *= 2.5
            phi *= 2.5
            s_loss = l2_loss(s_test_hat, s_test)
            # different = torch.sum(torch.norm(s_test-torch.ones_like(s_test),2))
            different_values.append(dis.item())
            u_loss_values.append(u_loss.item())
            s_loss_values.append(s_loss.item())
    different_values = np.array(different_values)
    # different_values = np.log(different_values)
    different_values = (different_values - different_values.min()) / (different_values.max() - different_values.min())
    u_loss_values = np.array(u_loss_values)
    s_loss_values = np.array(s_loss_values)

    # 创建画布
    plt.figure(figsize=size)

    sns.regplot(x=different_values, y=u_loss_values, scatter=False,
               ci=95, order=2, line_kws={'lw': 0.5})
    sns.regplot(x=different_values, y=s_loss_values, scatter=False,
            ci=95, order=2, line_kws={'lw': 0.5})

    sns.despine(top=True, right=True)

    output_path = '.\\result'
    import matplotlib.ticker as ticker

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    plt.xticks(fontsize=lablesize)
    plt.yticks(fontsize=lablesize)
    plt.tight_layout()
    plot_filename = os.path.join(output_path, 'loss_vs_different_domain.svg')
    plt.savefig(plot_filename, bbox_inches='tight')
    # 保存图表
    plt.show()


def vis_dis(test_loader,model_path,lablesize = 8,size=(2.65,2)):
    m = Model(embed_dim=128)
    m.load_state_dict(torch.load(model_path))
    m = m.to('cuda')
    m.eval()

    # Lists to store loss values
    u_loss_values = []
    s_loss_values = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Your existing data loading and processing code
            mag = batch.mag.to(device='cuda')
            period = batch.period.to(device='cuda')
            u_test = batch.u.to(device='cuda')
            phi = batch.phi.to(device='cuda')
            test_input = batch.test_input.to(device='cuda')
            model_input = test_input.reshape(-1, test_input.shape[1] * test_input.shape[2], test_input.shape[3])
            x_test = model_input[:, :, 0]
            t_test = model_input[:, :, 1]

            x_right = batch.x_right.to(device='cuda')
            s_t_test = batch.t.to(device='cuda')
            s_test = batch.s.to(device='cuda')

            phi_out_test, u_test_hat, _ = m.forward(t=t_test, mag=mag, period=period, x=x_test)
            phi_test = phi_out_test.reshape(u_test.shape)
            u_act = batch_interp(queries=phi_test, xs=phi, ys=u_test)
            u_test_hat = u_test_hat.reshape(u_act.shape)

            u_loss = torch.norm(u_test_hat.reshape(u_test_hat.shape[0],-1) - u_act.reshape(u_test_hat.shape[0],-1) , 2,dim=1) / torch.norm(u_test_hat.reshape(u_test_hat.shape[0],-1) , 2,dim=1)

            s_test_hat, _, _ = m.forward(t=s_t_test, mag=mag, period=period, x=x_right)
            x_hat = phi_out_test.reshape(u_test.shape)

            s_loss = torch.norm(s_test_hat - s_test, 2,dim=1) / torch.norm(s_test, 2,dim=1)

            # Store the loss values
            u_loss_values.append(u_loss.item())
            s_loss_values.append(s_loss.item())

    u_loss_values = np.array(u_loss_values)
    s_loss_values = np.array(s_loss_values)

    # Calculate means
    u_mean = u_loss_values.mean()
    s_mean = s_loss_values.mean()

    # Create figure with two subplots
    plt.figure(figsize=size)

    # Plot u_loss KDE
    plt.subplot(2, 1, 1)

    sns.kdeplot(u_loss_values, shade=True, color='blue')
    plt.axvline(x=u_mean, color='black', linestyle='--', linewidth=1)
    # plt.text(u_mean + 0.001, plt.ylim()[1] * 0.9, f'Mean: {u_mean:.4f}', color='black')
    # plt.xlabel('u_loss Value')
    # plt.title('u_loss Distribution')
    plt.gca().set_yticklabels([])  # Remove y-axis labels
    plt.gca().set_yticks([])  # Remove y-axis ticks
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=lablesize)  # You can adjust the size as needed

    # Plot s_loss KDE
    plt.subplot(2, 1, 2)
    sns.kdeplot(s_loss_values, shade=True, color='green')
    plt.axvline(x=s_mean, color='black', linestyle='--', linewidth=1)
    # plt.text(s_mean + 0.00001, plt.ylim()[1] * 0.9, f'Mean: {s_mean:.4f}', color='black')
    # plt.xlabel('s_loss Value')
    # plt.title('s_loss Distribution')
    plt.gca().set_yticklabels([])  # Remove y-axis labels
    plt.gca().set_yticks([])  # Remove y-axis ticks
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=lablesize)  # You can adjust the size as needed

    output_path = '.\\result'

    plt.tight_layout()
    plot_filename = os.path.join(output_path, 'stefan_dis.svg')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()

    # Print some statistics
    print(f"u_loss - Mean: {u_mean:.4f}, Std: {u_loss_values.std():.4f}")
    print(f"s_loss - Mean: {s_mean:.4f}, Std: {s_loss_values.std():.4f}")




def vis_field_comparison(test_loader, model_path1, model_path2,lablesize = 8,size=(2.65,2)):
    # Initialize models
    m1 = Model(embed_dim=128)
    m1.load_state_dict(torch.load(model_path1))
    m1 = m1.to('cuda')
    m1.eval()

    m2 = Model(embed_dim=128)
    m2.load_state_dict(torch.load(model_path2))
    m2 = m2.to('cuda')
    m2.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            mag = batch.mag.to(device='cuda')
            period = batch.period.to(device='cuda')
            test_input = batch.test_input.to(device='cuda')
            test_input = test_input[:, ::5, ::80, :]
            print(test_input.shape)

            # Prepare input data
            model_input = test_input.reshape(-1, test_input.shape[1] * test_input.shape[2], test_input.shape[3])
            x_test = model_input[:, :, 0]
            t_test = model_input[:, :, 1]

            # Process original coordinates
            def process_tensor(tensor):
                return tensor.squeeze().reshape(test_input.shape[1], test_input.shape[2]).cpu().numpy()

            t_test_np = process_tensor(t_test)
            x_np = process_tensor(x_test)

            # Get outputs from both models
            phi_out_test1, u_test_hat1, _ = m1.forward(t=t_test, mag=mag, period=period, x=x_test)
            phi_out_test2, u_test_hat2, _ = m2.forward(t=t_test, mag=mag, period=period, x=x_test)

            phi_hat_np1 = process_tensor(phi_out_test1)
            phi_hat_np2 = process_tensor(phi_out_test2)

            # Create first figure for model1
            plt.figure(figsize=size)
            plt.scatter(t_test_np, phi_hat_np1, c='k', s=0.05)
            for i in range(0, t_test_np.shape[0]):
                plt.plot(t_test_np[i, :], phi_hat_np1[i, :], 'k-', linewidth=0.1, alpha=0.5)
            for j in range(0, t_test_np.shape[1]):
                plt.plot(t_test_np[:, j], phi_hat_np1[:, j], 'k-', linewidth=0.1, alpha=0.5)
            plt.xticks([])  # Remove x-axis numbers
            plt.yticks([])  # Remove y-axis numbers
            plt.gca().set_frame_on(False)  # Turn off frame

            # Save first figure
            output_path = '.\\result'
            plot_filename1 = os.path.join(output_path, 'model1.png')
            plt.savefig(plot_filename1, dpi = 150, bbox_inches='tight')
            plt.close()

            # Create second figure for model2
            plt.figure(figsize=size)
            plt.scatter(t_test_np, phi_hat_np2, c='k', s=0.05)
            for i in range(0, t_test_np.shape[0]):
                plt.plot(t_test_np[i, :], phi_hat_np2[i, :], 'k-', linewidth=0.1, alpha=0.5)
            for j in range(0, t_test_np.shape[1]):
                plt.plot(t_test_np[:, j], phi_hat_np2[:, j], 'k-', linewidth=0.3, alpha=0.5)
            plt.xticks([])  # Remove x-axis numbers
            plt.yticks([])  # Remove y-axis numbers
            plt.gca().set_frame_on(False)  # Turn off frame

            # Save second figure
            plot_filename2 = os.path.join(output_path, 'model2.png')
            plt.savefig(plot_filename2,dpi = 150,  bbox_inches='tight')
            plt.close()

            break
def vis_result_t2(test_loader, model_path, plot_all = True,size = (11, 9),labelsize = 10):
    m = Model(embed_dim=128)
    m.load_state_dict(torch.load(model_path))
    m = m.to('cuda')

    case_indices = [2, 5, 14]  # 0-based indices for cases 3, 6, 15
    plot_types = ['Ground Truth', 'Prediction', 'Error']

    with torch.no_grad():
        if plot_all:
            fig = plt.figure(figsize=size)
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx not in case_indices:
                continue

            # Process the data
            mag = batch.mag.to(device='cuda')
            period = batch.period.to(device='cuda')
            u_test = batch.u.to(device='cuda')
            phi = batch.phi.to(device='cuda')
            test_input = batch.test_input.to(device='cuda')
            model_input = test_input.reshape(-1, test_input.shape[1] * test_input.shape[2], test_input.shape[3])
            x_test = model_input[:, :, 0]
            t_test = model_input[:, :, 1]
            t_real = batch.t.to(device='cuda')

            phi_out_test, u_test_hat, _ = m.forward(t=t_test, mag=mag, period=period, x=x_test)
            x_hat = phi_out_test.reshape(u_test.shape)
            phi_test = phi_out_test.reshape(u_test.shape)
            u_act = batch_interp(queries=phi_test, xs=phi, ys=u_test)
            u_test_hat = u_test_hat.reshape(u_act.shape)
            u_act *= 20
            u_test_hat *= 20
            loss = torch.abs(u_act - u_test_hat)
            u_real = u_act
            x_hat *= 2.5
            phi *= 2.5

            # Convert to numpy arrays
            u_test_hat = u_test_hat.squeeze(0).detach().cpu().numpy()
            t_real = t_real.squeeze(0).detach().cpu().numpy()
            t_real = t_real * total_time
            x_real = phi.squeeze(0).detach().cpu().numpy()
            x_hat = x_hat.squeeze(0).detach().cpu().numpy()
            u_real = u_real.squeeze(0).detach().cpu().numpy()
            loss = loss.squeeze(0).detach().cpu().numpy()

            # Determine which case we're processing (1, 2, or 3)
            case_num = case_indices.index(batch_idx) + 1

            # Create and save each plot separately
            for plot_type in plot_types:
                if not plot_all:
                    fig = plt.figure(figsize=(4, 3))
                    ax = fig.add_subplot(111)
                else:
                    ax = fig.add_subplot(3, 3, 3 * (case_num - 1) + plot_types.index(plot_type) + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_xlabel('Time')
                # ax.set_ylabel('Space')
                # ax.set_aspect('equal')  # This ensures equal aspect ratio

                ax.set_frame_on(False)
                # Common settings for all colorbars
                cbar_kwargs = {
                    'ax': ax,
                    'orientation': 'vertical',
                    'ticks': None,  # Will be set specifically for each case
                    'format': '%.2f'  # Format all numbers with 2 decimal places
                }

                if plot_type == 'Ground Truth':
                    im = ax.pcolormesh(t_real, x_real, u_real, cmap='turbo', shading='auto')
                    if case_num == 4:
                        cbar_kwargs['label'] = 'Temperature (°C)'
                    cbar = fig.colorbar(im,  ** cbar_kwargs)
                    # Set 4 evenly spaced ticks
                    ticks = np.linspace(np.min(u_real), np.max(u_real), 4)
                    cbar.set_ticks(ticks)
                    # Set font size
                    cbar.ax.tick_params(labelsize=labelsize)
                    if case_num == 4:
                        cbar.ax.yaxis.label.set_size(14)

                elif plot_type == 'Prediction':
                    im = ax.pcolormesh(t_real, x_hat, u_test_hat, cmap='turbo', shading='auto')
                    if case_num == 4:
                        cbar_kwargs['label'] = 'Temperature (°C)'
                    cbar = fig.colorbar(im,  ** cbar_kwargs)
                    # Set 4 evenly spaced ticks
                    ticks = np.linspace(np.min(u_test_hat), np.max(u_test_hat), 4)
                    cbar.set_ticks(ticks)
                    # Set font size
                    cbar.ax.tick_params(labelsize=labelsize)
                    if case_num == 4:
                        cbar.ax.yaxis.label.set_size(14)

                else:  # Error
                    im = ax.pcolormesh(t_real, x_hat, loss, cmap='turbo', shading='auto')
                    if case_num == 4:
                        cbar_kwargs['label'] = 'Absolute Square Error'
                    cbar = fig.colorbar(im,  ** cbar_kwargs)
                    # Set 4 evenly spaced ticks
                    ticks = np.linspace(np.min(loss), np.max(loss), 4)
                    cbar.set_ticks(ticks)
                    # Set font size
                    cbar.ax.tick_params(labelsize=labelsize)
                    if case_num == 4:
                        cbar.ax.yaxis.label.set_size(14)

                # ax.set_title(f'Case {case_num} - {plot_type}', pad=20)
                plt.tight_layout()

                # Save the figure
                if not plot_all:
                    output_path = '.\\result'
                    plot_filename = os.path.join(output_path,
                                                 f'stefan_case{case_num}_{plot_type.lower().replace(" ", "_")}.png')
                    plt.savefig(plot_filename, dpi=600, bbox_inches='tight')
                    plt.close()

        if plot_all:
            output_path = '.\\result'
            plot_filename = os.path.join(output_path,
                                         f'stefan_case_all.png')
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()

def visualize_phi_distribution(model_path, sample_size=10000):
    # Load the model
    # setup_seed(38600)
    m = Model(embed_dim=128)
    m.load_state_dict(torch.load(model_path))
    m = m.to('cuda')
    m.eval()
    train_loader = generate_data(batchsize=1, n_s=1, n_interior=sample_size,
                                 n_boundary=400, n_initial=400)
    # Generate some test data
    data_iter = iter(train_loader)
    batch = next(data_iter)
    # Get interior points
    interior = batch["interior"].to(device='cuda')
    interior_x_train = interior[:, :, 0]
    interior_t_train = interior[:, :, 1]
    mag = batch["mag"].to(device='cuda')
    period = batch["period"].to(device='cuda')
    period = torch.ones_like(period)*1.8
    mag = torch.ones_like(mag)*0.12
    # Forward pass to get phi values
    with torch.no_grad():
        phi_interior, u_real, _ = m.forward(t=interior_t_train, mag=mag, period=period, x=interior_x_train)

    t_vals = interior_t_train.cpu().numpy().flatten()
    phi_vals = phi_interior.cpu().numpy().flatten()
    interior_x_train = interior_x_train.cpu().numpy().flatten()

    # Create figure with two subplots
    plt.figure(figsize=(15, 6))

    # First subplot: Time vs Phase Field
    plt.subplot(1, 2, 1)
    plt.scatter(t_vals, phi_vals, alpha=0.5, s=5, c='blue')
    # plt.xlabel('Time (t)')
    # plt.ylabel('Spatial Field (x)')
    # plt.title('Physical Field')
    plt.xticks([])
    plt.yticks([])
    # plt.grid(True)
    plt.axis('off')

    # Second subplot: X vs Time
    plt.subplot(1, 2, 2)
    plt.scatter(t_vals,interior_x_train, alpha=0.5, s=5, c='red')
    # plt.xlabel('Time (t)')
    # plt.ylabel('Spatial Field (ξ)')
    # plt.title('Conjugate Field')
    plt.xticks([])
    plt.yticks([])
    # plt.grid(True)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("plot.png", dpi=600, bbox_inches='tight')

    plt.show()

def vis__boundary__dis(test_loader, model_path,size = (4,9),labelsize = 10):
    m = Model(embed_dim=128)
    m.load_state_dict(torch.load(model_path))
    m = m.to('cuda')
    m.eval()

    case_indices = [2, 5, 14]
    case_data = {idx: {'t': [], 'loss': []} for idx in case_indices}

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx not in case_indices:
                continue

            mag = batch.mag.to('cuda')
            period = batch.period.to('cuda')
            x_right = batch.x_right.to('cuda')
            s_t_test = batch.t.to('cuda')
            s_test = batch.s.to('cuda')

            s_test_hat, _, _ = m.forward(t=s_t_test, mag=mag, period=period, x=x_right)
            s_loss = torch.abs(s_test_hat - s_test)
            s_loss/= s_test

            t_cpu = s_t_test.cpu().numpy().squeeze()
            loss_cpu = s_loss.cpu().numpy().squeeze()

            case_data[batch_idx]['t'] = t_cpu
            case_data[batch_idx]['loss'] = loss_cpu

    plt.figure(figsize=size)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for idx, color in zip(case_indices, colors):
        sorted_indices = np.argsort(case_data[idx]['t'])
        t_sorted = case_data[idx]['t'][sorted_indices]
        loss_sorted = case_data[idx]['loss'][sorted_indices]

        ax = plt.subplot(3, 1, case_indices.index(idx) + 1)
        plt.plot(t_sorted, loss_sorted,
                 c=color,
                 alpha=0.7,
                 linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.yaxis.set_ticks_position('right')
        ax.tick_params(axis='y', which='both',
                       right=True,
                       labelright=True, labelsize=labelsize)

        # ax.grid(axis='y', alpha=0.3)

        plt.axhline(y=0, color='r', linestyle='--', alpha=0.4)

        # 美化标签和标题
        # plt.xlabel('Time step', fontsize=10)
        # plt.ylabel('Prediction Error', fontsize=10)
        plt.yscale('log')  # Add this line to set y-axis to logarithmic scale
        plt.xticks([])
        # if case_indices.index(idx) == 2:
        #     plt.xticks([0,1],[1,3600])
        # plt.grid(alpha=0.2)
        # plt.legend()

    plt.tight_layout()
    output_path = '.\\result'
    plot_filename = os.path.join(output_path,
                                 f's_dis.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    model_path = '.\goodmodel.pth'
    model_path2 = ".\\badmodel.pth"
    # visualize_phi_distribution(model_path)
    test_loader = create_dataloaders(".\stefan_data.npy",batch_size=1, shuffle=False,
                                     max_samples=300)
    print('start\n')
    vis_result_t2(test_loader,model_path,True,size=(6,4),labelsize=8)
    vis__boundary__dis(test_loader,model_path,size=(2,5),labelsize=8)
    vis_field_comparison(test_loader,model_path,model_path2,lablesize = 8,size=(2,1.25))
    vis_dis(test_loader, model_path,lablesize = 8,size=(2.5,1.45))
    vis_boundary_loss(test_loader,model_path,lablesize = 8,size=(2.65,2))