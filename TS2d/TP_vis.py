import matplotlib.pyplot as plt
import matplotlib
import os
import random
from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.Get_dataloader import create_data_loaders
from model.TP_model import TP_model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
def plot_error_distributions(
        model_path,
        data_loader,
        device=None,
        save_path=True,
        figsize=(6, 3),
        alpha=0.5
):
    setup_seed(42)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'  
    matplotlib.rcParams['mathtext.default'] = 'regular' 
    matplotlib.rcParams['axes.unicode_minus'] = True

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TP_model(num_layers=6, embed_dim=640)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    T_errors = []
    P_errors = []
    T_l2_errors = []
    P_l2_errors = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="computing"):
            x = batch['x'].to(device)
            phi = batch['phi'].to(device)
            T = batch['T'].to(device)
            P = batch['P'].to(device)
            fx = batch['fx'].to(device)
            shape = fx.shape[0]

            T_pred,P_pred = model(x=x, condition=fx)

            T_error = torch.mean(torch.abs(T_pred.reshape(shape, -1) - T.reshape(shape, -1)),dim=-1)
            T_errors.extend(T_error.cpu().numpy().flatten())

            p_error = torch.mean(torch.abs(P_pred.reshape(shape, -1) - P.reshape(shape, -1)),dim=-1)
            P_errors.extend(p_error.cpu().numpy().flatten())

            tl2_error = torch.norm(T_pred.reshape(shape, -1) - T.reshape(shape, -1), p=2) / torch.norm(
                T.reshape(shape, -1) + 1e-3, p=2)
            T_l2_errors.extend(tl2_error.cpu().numpy().flatten())

            pl2 = torch.norm(P_pred.reshape(shape, -1) - P.reshape(shape, -1), p=2) / torch.norm(
                P.reshape(shape, -1) + 1e-3, p=2)
            P_l2_errors.extend(pl2.cpu().numpy().flatten())


    T_errors = np.array(T_errors)
    P_errors = np.array(P_errors)
    T_l2_errors = np.array(T_l2_errors)
    P_l2_errors = np.array(P_l2_errors)
    T_mean = T_errors.mean()
    P_mean = P_errors.mean()
    T_l2_mean = T_l2_errors.mean()
    P_l2_mean = P_l2_errors.mean()

    print(f'abs T loss mean: {T_mean}')
    print(f'abs P loss mean: {P_mean}')
    print(f'T l2 loss mean: {T_l2_mean}')
    print(f'P l2 loss mean: {P_l2_mean}')
    plt.figure(figsize=figsize)
    plt.rcParams['xtick.labelsize'] = 14  # X-axis tick label size
    plt.rcParams['ytick.labelsize'] = 14

    sns.kdeplot(T_errors, fill=True, alpha=alpha, color='lightpink',linewidth=2, log_scale=True, label='Density')
    plt.axvline(T_mean, color='black', linestyle='--', linewidth=3)
    # plt.xlim(2e-3, 1e-1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    plt.tight_layout()
    plt.savefig('.\\T1.svg', bbox_inches='tight', transparent=True)
    plt.show()


    sns.kdeplot(P_errors, fill=True, alpha=alpha, color='silver',linewidth=2, log_scale=True, label='Density')
    plt.axvline(P_mean, color='black', linestyle='--', linewidth=3)
    # plt.xlim(2e-3, 1e-1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    plt.tight_layout()
    plt.savefig('.\\P1.svg', bbox_inches='tight', transparent=True)
    plt.show()


    sns.kdeplot(T_l2_errors, fill=True, alpha=alpha, color='sandybrown',linewidth=2, label='Density')
    plt.axvline(T_l2_mean, color='black', linestyle='--', linewidth=3)
    # plt.xlim(2e-3, 1e-1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    plt.tight_layout()
    plt.savefig('D:\\desktop\\output861\\vis_folder\\Tl2.svg', bbox_inches='tight', transparent=True)
    plt.show()

    sns.kdeplot(P_l2_errors, fill=True, alpha=alpha, color='paleturquoise', linewidth=2, label='Density')
    plt.axvline(P_l2_mean, color='black', linestyle='--', linewidth=3)
    # plt.xlim(2e-3, 1e-1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    # plt.tight_layout()
    plt.savefig('.\\Pl2.svg', bbox_inches='tight', transparent=True)
    plt.show()


def visualize_predictions_with_errors(model_path, data_loader, num_samples=5, save_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(42)
    m = TP_model(num_layers=6, embed_dim=512 + 128)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m = m.to(device)
    m.eval()

    batch = next(iter(data_loader))
    x = batch['x'].to(device)
    T = batch['T'].to(device)
    P = batch['P'].to(device)
    phi = batch['phi'].to(device)
    fx = batch['fx'].to(device)

    batch_size = x.shape[0]
    if num_samples > batch_size:
        num_samples = batch_size
        print(f"Warning: num_samples reduced to batch_size: {batch_size}")

    indices = random.sample(range(batch_size), num_samples)

    with torch.no_grad():
        T_pred, P_pred = m(x=x, condition=fx)

    phi_np = phi.cpu().numpy()
    T_pred_np = T_pred.cpu().numpy()
    P_pred_np = P_pred.cpu().numpy()
    T_np = T.cpu().numpy()
    P_np = P.cpu().numpy()

    def format_colorbar(cbar, vmin, vmax, n_ticks=5):
        ticks = np.linspace(vmin, vmax, n_ticks)
        tick_labels = []
        for tick in ticks:
            if abs(tick) < 0.01 or abs(tick) >= 1000:
                tick_labels.append(f"{tick:.2f}")
            elif abs(tick) < 1:
                tick_labels.append(f"{tick:.3f}")
            else:
                tick_labels.append(f"{tick:.2f}")

        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=12)

    n_ticks = 4
    for i, idx in enumerate(indices):
        fig = plt.figure(figsize=(20, 12))

        gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1])

        ax1 = fig.add_subplot(gs[0, 0])  
        ax2 = fig.add_subplot(gs[0, 1]) 
        ax3 = fig.add_subplot(gs[0, 2])  
        ax4 = fig.add_subplot(gs[0, 3])  
        ax5 = fig.add_subplot(gs[0, 4])
        ax6 = fig.add_subplot(gs[0, 5]) 

        phi_true = phi_np[idx]
        T_true = T_np[idx]
        P_true = P_np[idx]
        T_pred_sample = T_pred_np[idx]
        P_pred_sample = P_pred_np[idx]

        T_error = np.abs(T_true - T_pred_sample)
        P_error = np.abs(P_true - P_pred_sample)

        phi_true_x = phi_true[:, 0]
        phi_true_y = phi_true[:, 1]
        x_min, x_max = np.min(phi_true_x), np.max(phi_true_x)
        y_min, y_max = np.min(phi_true_y), np.max(phi_true_y)

        
        grid_resolution = 1000
        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        xi, yi = np.meshgrid(xi, yi)

        r_values = np.sqrt(phi_true_x ** 2 + phi_true_y ** 2)
        min_r = np.min(r_values)
        max_r = np.max(r_values)
        r_grid = np.sqrt(xi ** 2 + yi ** 2)
        ring_mask = (r_grid >= min_r) & (r_grid <= max_r)

        def interpolate_and_mask(data, method='cubic'):
            interp_data = griddata((phi_true_x, phi_true_y), data, (xi, yi),
                                   method=method, fill_value=np.nan)
            return np.where(ring_mask, interp_data, np.nan)

        T_true_interp = interpolate_and_mask(T_true)
        vmin_T_true = np.nanmin(T_true_interp)
        vmax_T_true = np.nanmax(T_true_interp)
        im1 = ax1.imshow(T_true_interp,
                         extent=[x_min, x_max, y_min, y_max],
                         origin='lower',
                         cmap='jet',
                         aspect='equal',
                         alpha=0.65,
                         vmin=vmin_T_true,
                         vmax=vmax_T_true)
        ax1.axis('off')
        ax1.set_aspect('equal')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        format_colorbar(cbar1, vmin_T_true, vmax_T_true, n_ticks)

        T_pred_interp = interpolate_and_mask(T_pred_sample)
        vmin_T_pred = np.nanmin(T_pred_interp)
        vmax_T_pred = np.nanmax(T_pred_interp)
        im2 = ax2.imshow(T_pred_interp,
                         extent=[x_min, x_max, y_min, y_max],
                         origin='lower',
                         cmap='jet',
                         aspect='equal',
                         alpha=0.65,
                         vmin=vmin_T_pred,
                         vmax=vmax_T_pred)
        ax2.axis('off')
        ax2.set_aspect('equal')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        format_colorbar(cbar2, vmin_T_pred, vmax_T_pred, n_ticks)

        T_error_interp = interpolate_and_mask(T_error)
        vmin_T_error = 0
        vmax_T_error = np.nanmax(T_error_interp)
        im3 = ax3.imshow(T_error_interp,
                         extent=[x_min, x_max, y_min, y_max],
                         origin='lower',
                         cmap='jet',
                         aspect='equal',
                         alpha=0.65,
                         vmin=vmin_T_error,
                         vmax=vmax_T_error)
        ax3.axis('off')
        ax3.set_aspect('equal')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        format_colorbar(cbar3, vmin_T_error, vmax_T_error, n_ticks)

        P_true_interp = interpolate_and_mask(P_true)
        vmin_P_true = np.nanmin(P_true_interp)
        vmax_P_true = np.nanmax(P_true_interp)
        im4 = ax4.imshow(P_true_interp,
                         extent=[x_min, x_max, y_min, y_max],
                         origin='lower',
                         cmap='Spectral',
                         aspect='equal',
                         alpha=0.95,
                         vmin=vmin_P_true,
                         vmax=vmax_P_true)
        ax4.axis('off')
        ax4.set_aspect('equal')
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        format_colorbar(cbar4, vmin_P_true, vmax_P_true, n_ticks)

        P_pred_interp = interpolate_and_mask(P_pred_sample)
        vmin_P_pred = np.nanmin(P_pred_interp)
        vmax_P_pred = np.nanmax(P_pred_interp)
        im5 = ax5.imshow(P_pred_interp,
                         extent=[x_min, x_max, y_min, y_max],
                         origin='lower',
                         cmap='Spectral',
                         aspect='equal',
                         alpha=0.95,
                         vmin=vmin_P_pred,
                         vmax=vmax_P_pred)
        ax5.axis('off')
        ax5.set_aspect('equal')
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        format_colorbar(cbar5, vmin_P_pred, vmax_P_pred, n_ticks)

        P_error_interp = interpolate_and_mask(P_error)
        vmin_P_error = 0
        vmax_P_error = np.nanmax(P_error_interp)
        im6 = ax6.imshow(P_error_interp,
                         extent=[x_min, x_max, y_min, y_max],
                         origin='lower',
                         cmap='Reds',
                         aspect='equal',
                         alpha=0.8,
                         vmin=vmin_P_error,
                         vmax=vmax_P_error)
        ax6.axis('off')
        ax6.set_aspect('equal')
        cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        format_colorbar(cbar6, vmin_P_error, vmax_P_error, n_ticks)


        plt.tight_layout()

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pdf_path = os.path.join(save_dir, f'TP_comparisonidx_{i:04d}.svg')
            # png_path = os.path.join(save_dir, f'TP_comparison_with_errors_{i + 1:03d}_idx_{idx:04d}.png')
            plt.savefig(pdf_path, bbox_inches='tight', format='svg')
            # plt.savefig(png_path, bbox_inches='tight', format='png', dpi=300)
            print(f"The pdf has been saved to: {pdf_path} ")

        # plt.show()
if __name__ == '__main__':
    seed = 42
    output_file_path = r"\TS_data_normalized.npz"
    uv_path = r"\TS_A_coord.npy"
    batchsize = 100

    from data.Get_dataloader import create_data_loaders

    train_loader, val_loader = create_data_loaders(output_file_path, uv_path, batch_size=batchsize, random_seed=seed)

    model_path = "\TPmodel.pth"
    save_viz_dir = "\TP_compare" 
    plot_error_distributions(
        model_path=model_path,
        data_loader=val_loader,

    )
