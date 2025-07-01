import os
from datetime import datetime
import math
import torch
from scipy.interpolate import interp1d
from sympy.benchmarks.bench_meijerint import alpha
from tensorboard.backend.event_processing import event_accumulator
import os
from data.supervise_train_data import load_supervise_data
from data.data import generate_data
from data.supervise_train_data import load_supervise_data
from model.TS_model import Model
from utils.loss import count_params, LpLoss, count_params_with_grad, setup_seed
from tensorboard.backend.event_processing import event_accumulator
from scipy.stats import norm
import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt

# train_data_path = "D:\\desktop\\BCO\\data\\test_data\\train_1000.npy"
# S_train_loader= load_supervise_data(batch_size=1, data_path=train_data_path,split_num=10)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        'truncated_' + cmap.name,
        cmap(np.linspace(minval, maxval, n))
    )

custom_cmap = truncate_colormap(plt.cm.Spectral, 0.1, 0.9)

myloss = torch.nn.MSELoss()
MSE_loss = torch.nn.MSELoss()
l2_loss = LpLoss(size_average=False)
l1 = LpLoss(p=2, size_average=True)
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
from typing import List, Optional, Dict




from torch.utils.data import Subset, DataLoader


def vis(test_loader, load_path=None,size = (4,9),labelsize = 10):
    m = Model(embed_dim=256)
    m.load_state_dict(torch.load(load_path))
    m = m.to('cuda')
    m.eval()

    with torch.no_grad():
        x_test_arr = []
        t_test_arr = []
        phi_hat_arr = []
        test_phi_arr = []
        T_hat_arr = []
        test_T_arr = []
        rho_hat_arr = []
        test_rho_arr = []
        v_hat_arr = []
        test_v_arr = []

        phi_erros_arr = []
        T_erros_arr = []
        rho_erros_arr = []
        v_erros_arr = []
        batch_idxs = [168, 44, 43]
        sample_indices = []
        for idx in batch_idxs:
            start = idx * test_loader.batch_size
            end = start + test_loader.batch_size
            sample_indices.extend(range(start, end))

        # 创建子集
        subset = Subset(test_loader.dataset, sample_indices)

        # 创建新DataLoader
        subset_loader = DataLoader(
            subset,
            batch_size=test_loader.batch_size,
            shuffle=False
        )
        # for batch_idx, batch in enumerate(test_loader[batch_idxs]):
        for batch in subset_loader:
            # if batch_idx >= 3:
            #     break
            test_points = batch["test_points"].to(device='cuda')
            test_T = batch["test_T"].to(device='cuda')
            test_rho = batch["test_rho"].to(device='cuda')
            test_v = batch["test_v"].to(device='cuda')
            test_phi = batch["test_phi"].to(device='cuda')
            heat_test = batch["heat_para"].to(device='cuda')

            x_test = test_points[:, :, 0]
            t_test = test_points[:, :, 1]

            phi_hat, T_hat, rho_hat, v_hat, urou, tao = m.forward(
                x=x_test, t=t_test, heat_para=heat_test)

            def process_tensor(tensor):
                return tensor.squeeze().reshape(100, 314).cpu().numpy()

            # Convert all tensors to numpy arrays with desired shape
            x_test_np = process_tensor(x_test)
            t_test_np = process_tensor(t_test)
            phi_hat_np = process_tensor(phi_hat)
            test_phi_np = process_tensor(test_phi)
            T_hat_np = process_tensor(T_hat)
            test_T_np = process_tensor(test_T)
            rho_hat_np = process_tensor(rho_hat)
            test_rho_np = process_tensor(test_rho)
            v_hat_np = process_tensor(v_hat)
            test_v_np = process_tensor(test_v)

            x_test_arr.append(x_test_np)
            t_test_arr.append(t_test_np)
            phi_hat_arr.append(phi_hat_np)
            test_phi_arr.append(test_phi_np)
            T_hat_arr.append(T_hat_np)
            test_T_arr.append(test_T_np)
            rho_hat_arr.append(rho_hat_np)
            test_rho_arr.append(test_rho_np)
            v_hat_arr.append(v_hat_np)
            test_v_arr.append(test_v_np)

            # Calculate absolute errors
            # phi_error = np.abs(phi_hat_np - test_phi_np)/(np.abs(phi_hat_np)+1e-2)
            # T_error = np.abs(T_hat_np - test_T_np)/(np.abs(test_T_np)+1e-2)
            # rho_error = np.abs(rho_hat_np - test_rho_np)/(np.abs(test_rho_np)+1e-2)
            # v_error = np.abs(v_hat_np - test_v_np)/(np.abs(test_v_np)+1e-2)
            phi_error = np.abs(phi_hat_np - test_phi_np)
            T_error = np.abs(T_hat_np - test_T_np)
            rho_error = np.abs(rho_hat_np - test_rho_np)
            v_error = np.abs(v_hat_np - test_v_np)

            phi_erros_arr.append(phi_error)
            T_erros_arr.append(T_error)
            rho_erros_arr.append(rho_error)
            v_erros_arr.append(v_error)

        arrs = [[test_T_arr,
                 test_rho_arr,
                 test_v_arr,
                 test_phi_arr],
                [T_hat_arr,
                 rho_hat_arr,
                 v_hat_arr,
                 phi_hat_arr],
                [T_erros_arr,
                 rho_erros_arr,
                 v_erros_arr,
                 phi_erros_arr]]
        # arrs = np.array(arrs)
        # 计算全局归一化范围
        global_norms = []
        for data_group in [arrs[0], arrs[1], arrs[2]]:  # GT, Pred, Error
            group_norms = []
            for i in range(4):  # 4个变量
                all_data = np.concatenate([arr for batch in data_group[i] for arr in batch])
                vmin, vmax = np.min(all_data), np.max(all_data)
                group_norms.append(Normalize(vmin=vmin, vmax=vmax))
            global_norms.append(group_norms)

        # Create figure with GridSpec
        fig = plt.figure(figsize=size, facecolor='none')  # Increased width to accommodate individual colorbars
        # Grid layout: 3 rows (data), 8 columns (4 plots + 4 colorbars)
        gs = fig.add_gridspec(3, 8, width_ratios=[1, 0.1, 1, 0.1, 1, 0.1, 1, 0.05])

        plot_axes = []
        cbar_axes = []

        for row in range(3):
            row_plot_axes = []
            row_cbar_axes = []
            for col in range(4):
                # Main plot axes - placed in columns 0, 2, 4, 6
                ax = fig.add_subplot(gs[row, col * 2], projection='3d', )
                row_plot_axes.append(ax)
                # ax.spines['top'].set_visible(False)
                # ax.spines['right'].set_visible(False)
                # ax.spines['bottom'].set_visible(False)
                # ax.spines['left'].set_visible(False)
                ax.set_axis_off()  # 这比单独设置spines和ticks更彻底
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.grid(False)

                # Colorbar axes - placed in columns 1, 3, 5, 7
                cax = fig.add_subplot(gs[row, col * 2 + 1])
                row_cbar_axes.append(cax)
                cax.set_frame_on(False)

            plot_axes.append(row_plot_axes)
            cbar_axes.append(row_cbar_axes)

        # ims = []
        # 核心修改：逐图独立归一化
        for y in range(len(test_T_arr)):  # 遍历每个batch
            for row in range(3):  # GT/Pred/Error三行
                for col in range(4):  # 温度/密度/速度/粒子四列
                    print(f'{y} {row} {col}\n')
                    data = arrs[row][col][y]

                    vmin = np.min(data)
                    vmax = np.max(data)
                    norm = Normalize(vmin, vmax)

                    y_plane = np.full_like(t_test_arr[y], y * 2.0)

                    if col == 3:  # 粒子追踪χ
                        Z = x_test_arr[y]
                    else:
                        Z = test_phi_arr[y]

                    colors = plt.cm.Spectral(norm(data))
                    alphas = [1.0, 0.8, 0.6]
                    im = plot_axes[row][col].plot_surface(
                        t_test_arr[y],
                        y_plane,
                        Z,
                        facecolors=colors,
                        rstride=1,
                        cstride=1,
                        antialiased=True,
                        shade=False,
                    )
                    plot_axes[row][col].view_init(elev=30, azim=-45)

                    if y == len(test_T_arr) - 1:

                        plot_axes[row][col].set_position([
                            plot_axes[row][col].get_position().x0,
                            plot_axes[row][col].get_position().y0 + plot_axes[row][
                                col].get_position().height * 0.3 * row,
                            plot_axes[row][col].get_position().width,
                            plot_axes[row][col].get_position().height
                        ])

                        sm = ScalarMappable(cmap=plt.cm.Spectral, norm=norm)
                        sm.set_array([])
                        if col != 3:
                            cbar_axes[row][col].set_position([
                                cbar_axes[row][col].get_position().x0 - cbar_axes[row][col].get_position().width,
                                cbar_axes[row][col].get_position().y0 + cbar_axes[row][col].get_position().height * (
                                            0.1 + 0.3 * row),
                                cbar_axes[row][col].get_position().width* 0.5,
                                cbar_axes[row][col].get_position().height * 0.65
                            ])
                        else:
                            cbar_axes[row][col].set_position([
                                cbar_axes[row][col].get_position().x0 - cbar_axes[row][col].get_position().width * 2,
                                cbar_axes[row][col].get_position().y0 + cbar_axes[row][col].get_position().height * (
                                            0.1+ 0.3 * row),
                                cbar_axes[row][col].get_position().width,
                                cbar_axes[row][col].get_position().height * 0.65
                            ])
                        cbar = fig.colorbar(sm, cax=cbar_axes[row][col], orientation='vertical')
                        tick_values = np.linspace(vmin, vmax, 4)  # 创建4个均匀分布的刻度值
                        cbar.set_ticks(tick_values)  # 设置刻度位置
                        cbar.set_ticklabels([f"{x:.3f}" for x in tick_values])  # 设置刻度标签，保留两位小数
                        cbar.ax.tick_params(labelsize=labelsize)

        output_path = '.\\result'
        plot_filename = os.path.join(output_path, 'rob.png')
        # 固定视角

        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', transparent=True)
        print("saved\n")
        # plt.show()
        for i in range(20):
            plt.close()


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def vis_time_dis2(test_loader, load_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = Model(embed_dim=256)
    m.load_state_dict(torch.load(load_path, map_location=device))
    m = m.to(device)
    m.eval()

    # Initialize storage dictionaries for each physical quantity
    error_dicts = {
        'phi': {},
        'v': {},
        'T': {},
        'rho': {}
    }
    count_dict = {}  # To record sample count at each time point

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            test_points = batch["test_points"].to(device)
            test_T = batch["test_T"].to(device)
            test_rho = batch["test_rho"].to(device)
            test_v = batch["test_v"].to(device)
            test_phi = batch["test_phi"].to(device)
            heat_test = batch["heat_para"].to(device)

            x_test = test_points[:, :, 0]
            t_test = test_points[:, :, 1]  # Shape: [batch_size, num_points]

            phi_hat, T_hat, rho_hat, v_hat, _, _ = m.forward(
                x=x_test, t=t_test, heat_para=heat_test)

            # Calculate absolute errors for each physical quantity
            errors = {
                'phi': torch.abs(phi_hat - test_phi).cpu().numpy(),
                'v': torch.abs(v_hat - test_v).cpu().numpy(),
                'T': torch.abs(T_hat - test_T).cpu().numpy(),
                'rho': torch.abs(rho_hat - test_rho).cpu().numpy()
            }

            # Get time data
            batch_times_np = t_test.cpu().numpy().flatten()

            # Store errors by time point
            for t_idx, t in enumerate(batch_times_np):
                if t not in count_dict:
                    count_dict[t] = 0
                    for phy in error_dicts:
                        error_dicts[phy][t] = []

                count_dict[t] += 1
                for phy in error_dicts:
                    error_dicts[phy][t].append(errors[phy].flatten()[t_idx])

    # Prepare plot data
    times = np.array(sorted(count_dict.keys()))

    # Calculate mean and std for each physical quantity
    plot_data = {}
    for phy in error_dicts:
        means = np.array([np.mean(error_dicts[phy][t]) for t in times])
        stds = np.array([np.std(error_dicts[phy][t]) for t in times])
        counts = np.array([count_dict[t] for t in times])

        # Create smooth curves using spline interpolation
        if len(times) > 3:  # Need at least 4 points for spline
            # Create finer grid for smooth curves
            smooth_times = np.linspace(times.min(), times.max(), 300)

            # Smooth mean values
            spl_mean = make_interp_spline(times, means, k=3)
            smooth_means = spl_mean(smooth_times)

            # Smooth std values
            spl_std = make_interp_spline(times, stds, k=3)
            smooth_stds = spl_std(smooth_times)

            plot_data[phy] = {
                'times': smooth_times,
                'mean': smooth_means,
                'std': smooth_stds,
                'count': counts
            }
        else:
            plot_data[phy] = {
                'times': times,
                'mean': means,
                'std': stds,
                'count': counts
            }

    # Set up the plot
    plt.figure(figsize=(12, 8))

    colors = {
        'phi': 'blue',
        'v': 'green',
        'T': 'red',
        'rho': 'purple'
    }

    # Plot each physical quantity with smooth line and shaded uncertainty band
    for phy in plot_data:
        # Main smooth line
        plt.plot(plot_data[phy]['times'], plot_data[phy]['mean'],
                 label=f'{phy} error', color=colors[phy], linewidth=2)

        # Shaded uncertainty band
        plt.fill_between(plot_data[phy]['times'],
                         plot_data[phy]['mean'] - plot_data[phy]['std'],
                         plot_data[phy]['mean'] + plot_data[phy]['std'],
                         color=colors[phy], alpha=0.2)

    # Set logarithmic scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Time (log scale)')
    plt.ylabel('Mean Absolute Error ± Std Dev (log scale)')
    plt.title('Error Distribution Over Time for Different Physical Quantities')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Add sample count on secondary axis (optional)
    if len(times) > 3:
        # Smooth the sample count if enough points
        spl_count = make_interp_spline(times, plot_data['phi']['count'], k=3)
        smooth_counts = spl_count(plot_data['phi']['times'])
        ax2 = plt.gca().twinx()
        ax2.plot(plot_data['phi']['times'], smooth_counts, 'k--', alpha=0.3, label='Sample count')
        ax2.set_ylabel('Sample Count')
        ax2.legend(loc='upper right')
    else:
        ax2 = plt.gca().twinx()
        ax2.plot(times, plot_data['phi']['count'], 'k--', alpha=0.3, label='Sample count')
        ax2.set_ylabel('Sample Count')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('physical_quantities_error_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def vis_time_dis(test_loader, load_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = Model(embed_dim=256)
    m.load_state_dict(torch.load(load_path, map_location=device))
    m = m.to(device)
    m.eval()

    all_times = []
    errors_phi = []
    errors_v = []
    errors_T = []
    errors_rho = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            test_points = batch["test_points"].to(device)
            test_T = batch["test_T"].to(device)
            test_rho = batch["test_rho"].to(device)
            test_v = batch["test_v"].to(device)
            test_phi = batch["test_phi"].to(device)
            heat_test = batch["heat_para"].to(device)

            x_test = test_points[:, :, 0]
            t_test = test_points[:, :, 1]  # Shape: [batch_size, num_points]

            unique, inverse_indices = torch.unique(t_test[0], return_inverse=True)
            unique_t, _ = torch.sort(unique)
            a = unique_t - unique
            time_steps = len(unique_t)
            num = x_test.shape[1]
            num_per_time = num // time_steps

            phi_hat, T_hat, rho_hat, v_hat, _, _ = m.forward(
                x=x_test, t=t_test, heat_para=heat_test)

            batch_size0 = test_T.shape[0]
            phi_hat = phi_hat.reshape(batch_size0, time_steps, num_per_time)
            T_hat = T_hat.reshape(batch_size0, time_steps, num_per_time)
            rho_hat = rho_hat.reshape(batch_size0, time_steps, num_per_time)
            v_hat = v_hat.reshape(batch_size0, time_steps, num_per_time)

            test_phi = test_phi.reshape(batch_size0, time_steps, num_per_time)
            test_T = test_T.reshape(batch_size0, time_steps, num_per_time)
            test_rho = test_rho.reshape(batch_size0, time_steps, num_per_time)
            test_v = test_v.reshape(batch_size0, time_steps, num_per_time)

            for i in range(time_steps):
                t_val = unique_t[i].item()

                phi_denom = torch.norm(phi_hat[:, i, :], p=2)
                v_denom = torch.norm(v_hat[:, i, :], p=2)
                T_denom = torch.norm(T_hat[:, i, :], p=2)
                rho_denom = torch.norm(rho_hat[:, i, :], p=2)

                # For values less than 5e-1, set to 5e-1
                if phi_denom < 5e-1: phi_denom = torch.tensor(5e-1).to(device)
                if v_denom < 5e-1: v_denom = torch.tensor(5e-1).to(device)
                if T_denom < 5e-1: T_denom = torch.tensor(5e-1).to(device)
                if rho_denom < 5e-1: rho_denom = torch.tensor(5e-1).to(device)

                # Calculate relative errors with L1 norm
                phi_err = (torch.norm(phi_hat[:, i, :] - test_phi[:, i, :], p=2) /
                           phi_denom).item()
                v_err = (torch.norm(v_hat[:, i, :] - test_v[:, i, :], p=2) /
                         v_denom).item()
                T_err = (torch.norm(T_hat[:, i, :] - test_T[:, i, :], p=2) /
                         T_denom).item()
                rho_err = (torch.norm(rho_hat[:, i, :] - test_rho[:, i, :], p=2) /
                           rho_denom).item()


                # Calculate relative errors with L1 norm
                # phi_err =  (torch.mean(torch.abs(phi_hat[:, i, :] - test_phi[:, i, :]),dim=-1)/
                #            torch.mean(torch.abs(test_phi[:, i, :]),dim=-1)).item()
                # v_err = (torch.mean(torch.abs(v_hat[:, i, :] - test_v[:, i, :]), dim=-1) /
                #            torch.mean(torch.abs(test_v[:, i, :]), dim=-1)).item()
                # T_err = (torch.mean(torch.abs(T_hat[:, i, :] - test_T[:, i, :]), dim=-1) /
                #            torch.mean(torch.abs(test_T[:, i, :]), dim=-1)).item()
                # rho_err = (torch.mean(torch.abs(rho_hat[:, i, :] - test_rho[:, i, :]), dim=-1) /
                #            torch.mean(torch.abs(test_rho[:, i, :]), dim=-1)).item()


                all_times.append(t_val)
                errors_phi.append(phi_err)
                errors_v.append(v_err)
                errors_T.append(T_err)
                errors_rho.append(rho_err)

    all_times = np.array(all_times)
    errors_phi = np.array(errors_phi)
    errors_v = np.array(errors_v)
    errors_T = np.array(errors_T)
    errors_rho = np.array(errors_rho)

    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 10))

    import pandas as pd
    error_data = pd.DataFrame({
        'Time': np.tile(all_times, 4),
        'Relative Error': np.concatenate([errors_phi, errors_v, errors_T, errors_rho]),
        'Variable': ['φ '] * len(all_times) +
                    ['v '] * len(all_times) +
                    ['T '] * len(all_times) +
                    ['ρ'] * len(all_times)
    })
    window_size = 10
    error_data['Smoothed Error'] = error_data.groupby('Variable')['Relative Error'].transform(
        lambda x: x.rolling(window_size, min_periods=1).mean()
    )

    sns.lineplot(
        data=error_data,
        x='Time',
        y='Smoothed Error',
        hue='Variable',
        errorbar=('ci', 95),
        err_kws={'alpha': 0.2},
        n_boot=5000,
        linewidth=1.5,
        legend=False
    )


    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.yscale('log')

    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = '.\\result'
    plot_filename = os.path.join(output_path, 'robtimedis.svg')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()


def vis_dis(test_loader, load_path=None,size = (4,9),labelsize = 10):
    m = Model(embed_dim=256)
    m.load_state_dict(torch.load(load_path))
    m = m.to('cuda')
    m.eval()

    # Initialize lists to store relative errors for each batch
    phi_erros_arr = []
    v_erros_arr = []
    T_erros_arr = []
    rho_erros_arr = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            test_points = batch["test_points"].to(device='cuda')
            test_T = batch["test_T"].to(device='cuda')
            test_rho = batch["test_rho"].to(device='cuda')
            test_v = batch["test_v"].to(device='cuda')
            test_phi = batch["test_phi"].to(device='cuda')
            heat_test = batch["heat_para"].to(device='cuda')

            x_test = test_points[:, :, 0]
            t_test = test_points[:, :, 1]

            phi_hat, T_hat, rho_hat, v_hat, urou, tao = m.forward(
                x=x_test, t=t_test, heat_para=heat_test)

            # Calculate relative errors for this batch
            phi_rel_error = (torch.norm(phi_hat - test_phi, 2) / torch.norm(test_phi, 2)).cpu().numpy().flatten()
            v_rel_error = (torch.norm(v_hat - test_v, 2) / torch.norm(test_v, 2)).cpu().numpy().flatten()
            T_rel_error = (torch.norm(T_hat - test_T, 2) / torch.norm(test_T, 2)).cpu().numpy().flatten()
            rho_rel_error = (torch.norm(rho_hat - test_rho, 2) / torch.norm(test_rho, 2)).cpu().numpy().flatten()

            # Append to lists
            phi_erros_arr.extend(phi_rel_error)
            v_erros_arr.extend(v_rel_error)
            T_erros_arr.extend(T_rel_error)
            rho_erros_arr.extend(rho_rel_error)

    # Convert to numpy arrays for plotting
    phi_erros_arr = np.array(phi_erros_arr)
    v_erros_arr = np.array(v_erros_arr)
    T_erros_arr = np.array(T_erros_arr)
    rho_erros_arr = np.array(rho_erros_arr)

    # phi_erros_arr = np.log10(phi_erros_arr)
    # v_erros_arr = np.log10(v_erros_arr)
    # T_erros_arr = np.log10(T_erros_arr)
    # rho_erros_arr = np.log10(rho_erros_arr)

    phi_rho_max = 0.008
    phi_rho_min = 0
    v_T_max = 0.1
    v_T_min = 0

    # phi_rho_max = np.max(np.concatenate([phi_erros_arr, rho_erros_arr]))
    # phi_rho_min = np.min(np.concatenate([phi_erros_arr, rho_erros_arr]))
    # v_T_max = np.max(np.concatenate([v_erros_arr, T_erros_arr]))
    # v_T_min = np.min(np.concatenate([v_erros_arr, T_erros_arr]))
    print(f'{phi_rho_min}\n')
    print(f'{np.min(v_erros_arr)}')

    # Calculate mean values
    phi_mean = np.mean(phi_erros_arr)
    v_mean = np.mean(v_erros_arr)
    T_mean = np.mean(T_erros_arr)
    rho_mean = np.mean(rho_erros_arr)

    # Create a figure with 4 subplots
    fig = plt.figure(figsize=(8, 9), facecolor='none')
    fig.patch.set_alpha(0)
    plt.rcParams['xtick.labelsize'] = 20  # X-axis tick label size
    plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick label size
    # Define different colors for each plot
    colors = ['blue', 'green', 'orange', 'red']

    # Phi plot
    ax = plt.subplot(4, 1, 1)
    ax.patch.set_alpha(0)
    sns.kdeplot(phi_erros_arr, fill=True, color=colors[0], alpha=0.3, log_scale=True)
    plt.axvline(phi_mean, color='black', linestyle='--', linewidth=3)
    # plt.text(phi_mean, plt.ylim()[1] * 0.9,
    #          horizontalalignment='center', color='black')
    # plt.title(r'$\chi$ Relative Error Distribution')
    # plt.xlabel('Relative Error')
    plt.ylabel('')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # plt.xscale('log')
    plt.xticks([])
    plt.yticks([])  # Remove y-axis values
    plt.xlim(phi_rho_min, phi_rho_max)
    ax.set_position([
        ax.get_position().x0,
        ax.get_position().y0,
        ax.get_position().width,
        ax.get_position().height
    ])

    # V plot
    ax = plt.subplot(4, 1, 2)
    ax.patch.set_alpha(0)
    sns.kdeplot(v_erros_arr, fill=True, color=colors[1], alpha=0.3, log_scale=True)
    plt.axvline(v_mean, color='black', linestyle='--', linewidth=3)
    # plt.text(v_mean, plt.ylim()[1] * 0.9, f'Mean: {v_mean:.4f}',
    #          horizontalalignment='center', color='red')
    # plt.title(r'$v$ Relative Error Distribution')
    # plt.xlabel('Relative Error')
    plt.ylabel('')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # plt.xscale('log')
    plt.xticks([])
    plt.yticks([])  # Remove y-axis values
    plt.xlim(v_T_min, v_T_max)
    ax.set_position([
        ax.get_position().x0,
        ax.get_position().y0 - ax.get_position().height * 1 / 4,
        ax.get_position().width,
        ax.get_position().height
    ])

    # T plot
    ax = plt.subplot(4, 1, 3)
    ax.patch.set_alpha(0)
    sns.kdeplot(T_erros_arr, fill=True, color=colors[2], alpha=0.3, log_scale=True)
    plt.axvline(T_mean, color='black', linestyle='--', linewidth=3)
    # plt.text(T_mean, plt.ylim()[1] * 0.9, f'Mean: {T_mean:.4f}',
    #          horizontalalignment='center', color='red')
    # plt.title('T Relative Error Distribution')
    # plt.xlabel('Relative Error')
    plt.ylabel('')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # plt.xscale('log')
    # plt.xticks([])
    plt.yticks([])  # Remove y-axis values
    plt.xlim(v_T_min, v_T_max)
    ax.set_position([
        ax.get_position().x0,
        ax.get_position().y0 + ax.get_position().height * 3 / 4,
        ax.get_position().width,
        ax.get_position().height
    ])

    # Rho plot
    ax = plt.subplot(4, 1, 4)
    ax.patch.set_alpha(0)
    sns.kdeplot(rho_erros_arr, fill=True, color=colors[3], alpha=0.3, log_scale=True)
    plt.axvline(rho_mean, color='black', linestyle='--', linewidth=3)
    # plt.text(rho_mean, plt.ylim()[1] * 0.9, f'Mean: {rho_mean:.4f}',
    #          horizontalalignment='center', color='red')
    # plt.title(r'$\rho$ Relative Error Distribution')
    # plt.xlabel('Relative Error')
    plt.ylabel('')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # plt.xscale('log')
    plt.yticks([])  # Remove y-axis values
    plt.xlim(phi_rho_min, phi_rho_max)
    ax.set_position([
        ax.get_position().x0,
        ax.get_position().y0 + ax.get_position().height * 13 / 4,
        ax.get_position().width,
        ax.get_position().height
    ])

    # plt.tight_layout()
    output_path = '.\\result'
    plot_filename = os.path.join(output_path, 'robdis.svg')
    plt.savefig(plot_filename,  bbox_inches='tight')
    plt.show()

    # Print average metrics
    print("\nFinal Average Relative Errors:")
    print(f"Phi - Mean: {phi_mean:.4f}")
    print(f"V - Mean: {v_mean:.4f}")
    print(f"T - Mean: {T_mean:.4f}")
    print(f"Rho - Mean: {rho_mean:.4f}")


def gettest(test_loader, load_path=None):
    m = Model(embed_dim=256)

    m.load_state_dict(torch.load(load_path))
    m = m.to('cuda')
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
            heat_test = batch["heat_para"].to(device='cuda')

            x_test = test_points[:, :, 0]
            t_test = test_points[:, :, 1]

            phi_hat, T_hat, rho_hat, v_hat, urou, tao = m.forward(
                x=x_test, t=t_test, heat_para=heat_test)

            phi_loss_l2 = l2_loss(phi_hat, test_phi)
            phi_loss_mse = MSE_loss(phi_hat, test_phi)
            v_loss_l2 = l2_loss(v_hat, test_v)
            v_loss_mse = MSE_loss(v_hat, test_v)
            rho_loss_l2 = l2_loss(rho_hat, test_rho)
            rho_loss_mse = MSE_loss(rho_hat, test_rho)
            T_loss_l2 = l2_loss(T_hat, test_T)
            T_loss_mse = MSE_loss(T_hat, test_T)
            phi_l2 += phi_loss_l2
            phi_mse += phi_loss_mse
            v_l2 += v_loss_l2
            v_mse += v_loss_mse
            T_l2 += T_loss_l2
            T_mse += T_loss_mse
            rho_l2 += rho_loss_l2
            rho_mse += rho_loss_mse
    phi_l2 = phi_l2 / len(test_loader)
    phi_mse = phi_mse / len(test_loader)
    v_l2 = v_l2 / len(test_loader)
    v_mse = v_mse / len(test_loader)
    rho_l2 = rho_l2 / len(test_loader)
    rho_mse = rho_mse / len(test_loader)
    T_l2 = T_l2 / len(test_loader)
    T_mse = T_mse / len(test_loader)
    print("\nFinal Average Losses:")
    print(f"Phi - Avg L2: {phi_l2.item():.4f}, Avg MSE: {phi_mse.item():.4f}")
    print(f"V - Avg L2: {v_l2.item():.4f}, Avg MSE: {v_mse.item():.4f}")
    print(f"T - Avg L2: {T_l2.item():.4f}, Avg MSE: {T_mse.item():.4f}")
    print(f"Rho - Avg L2: {rho_l2.item():.4f}, Avg MSE: {rho_mse.item():.4f}")



if __name__ == '__main__':
    test_data_path = ".\\zero_shoot.npy"
    # test_data_path = ".\\test_200.npy"

    test_loader = load_supervise_data(batch_size=1, data_path=test_data_path, split_num=200)
    load_path = ".\\PI_model.pth"
    vis(test_loader,load_path=load_path,size = (6,5),labelsize = 5)

    # vis_time_dis(test_loader, load_path)
    # vis_dis(test_loader, load_path,size = (8,8),labelsize = 6)

