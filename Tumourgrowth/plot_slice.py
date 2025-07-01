import math

import os
import numpy as np
from matplotlib.tri import Triangulation  # Add this import at the top of your file
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.path import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.value_test_data import create_dataloaders
from model.Whole_model import Model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        'truncated_' + cmap.name,
        cmap(np.linspace(minval, maxval, n))
    )

custom_cmap = truncate_colormap(plt.cm.rainbow, 0.15, 0.85)
# custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors, N=256)



test_path = ".\\all_test.npy"

test_loader = create_dataloaders(test_path, batch_size=1,shuffle = False, max_samples=200)

# 示例用法
myloss = LpLoss(size_average=False)
MSE_loss = torch.nn.HuberLoss()
l2_loss = LpLoss(size_average=False)
l2_loss_train = LpLoss(size_average=False)


def val(embed_dim=64, seed=420, load_path=None, output_dir="results"):
    setup_seed(seed)
    m = Model(embed_dim=embed_dim, pretrain_path='.\\pretrainmodel_posttrained.pth')
    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}. Check model architecture or compatibility. Training from scratch.")
    m = m.to('cuda')
    a = count_params_with_grad(m)
    print(a)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_sup = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx > 10:
                break

            value_test = batch.value.to(device)
            coord_test = batch.coord.to(device)
            s1_test = batch.s1.to(device)
            initial_domain_test = batch.initial_domain.to(device)
            mask_test = batch.mask.to(device)  # [1,15,128,128]
            contour = batch.processed_contour.to(device)
            contour = contour[:, :, :, :2]  # [1,15,300,2]

            x_in, y_in, t_in = coord_test[:, :, :, :, 0], coord_test[:, :, :, :, 1], coord_test[:, :, :, :, 2]
            x_in = x_in * mask_test.float()
            y_in = y_in * mask_test.float()
            t_in = t_in * mask_test.float()

            x_test = x_in.reshape(x_in.shape[0], x_in.shape[1] * x_in.shape[2] * x_in.shape[3])
            y_test = y_in.reshape(x_test.shape)
            t_test = t_in.reshape(x_test.shape)

            u_hat_test = m.forward(x=x_test, y=y_test, t=t_test, init_domain=initial_domain_test, s1=s1_test)
            u_hat_test = u_hat_test.reshape(mask_test.shape)
            masked_u_hat = u_hat_test * mask_test.float()
            masked_value = value_test * mask_test.float()
            loss_sup_test = l2_loss(masked_u_hat, masked_value)
            print(loss_sup_test)
            test_sup += loss_sup_test
            # value_test = torch.sin(80*(1-value_test))
            # u_hat_test = torch.sin(80*(1-u_hat_test))
            value_np = value_test.cpu().numpy()  # [1,15,128,128]
            u_hat_np = u_hat_test.cpu().numpy()  # [1,15,128,128]
            mask_np = mask_test.cpu().numpy()
            x_np = x_in.cpu().numpy()  # [1,15,128,128]
            y_np = y_in.cpu().numpy()  # [1,15,128,128]
            abs_error = np.abs(value_np - u_hat_np)

            # Create directory for this batch
            batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
            os.makedirs(batch_dir, exist_ok=True)

            # Process each sample in the batch
            for sample_idx in range(value_np.shape[0]):
                sample_dir = os.path.join(batch_dir, f"sample_{sample_idx}")
                os.makedirs(sample_dir, exist_ok=True)

                # Create a single figure for all slices of this sample
                fig = plt.figure(figsize=(10, 4.5), facecolor='none', edgecolor='none')  # Transparent figure
                gs = fig.add_gridspec(3, 5, width_ratios=[1] * 5)  # 3 rows (truth, pred, error), 10 columns (slices)

                # Select 10 slices (from 5 to 14)
                selected_slices = range(5,10)

                for col, slice_idx in enumerate(selected_slices):
                    # Get data for this slice
                    value_slice = value_np[sample_idx, slice_idx]
                    pred_slice = u_hat_np[sample_idx, slice_idx]
                    error_slice = abs_error[sample_idx, slice_idx]
                    mask_slice = mask_np[sample_idx, slice_idx]
                    x_slice = x_np[sample_idx, slice_idx]
                    y_slice = y_np[sample_idx, slice_idx]

                    # Create masked versions
                    masked_x = x_slice[mask_slice]
                    masked_y = y_slice[mask_slice]
                    masked_value = value_slice[mask_slice]
                    masked_pred = pred_slice[mask_slice]
                    masked_error = error_slice[mask_slice]

                    # Combine with contour points
                    current_contour = contour[sample_idx, slice_idx].cpu().numpy()
                    combined_x = np.concatenate([masked_x, current_contour[:, 0]])
                    combined_y = np.concatenate([masked_y, current_contour[:, 1]])

                    # Create values for contour points
                    contour_value = value_slice[mask_slice].mean() * np.ones(len(current_contour))
                    contour_pred = pred_slice[mask_slice].mean() * np.ones(len(current_contour))
                    contour_error = error_slice[mask_slice].mean() * np.ones(len(current_contour))

                    # Combine values
                    combined_value = np.concatenate([masked_value, contour_value])
                    combined_pred = np.concatenate([masked_pred, contour_pred])
                    combined_error = np.concatenate([masked_error, contour_error])

                    # Create triangulation
                    triang = Triangulation(combined_x, combined_y)
                    contour_path = Path(current_contour)

                    # Calculate triangle centers and create mask
                    triangles = triang.triangles
                    tri_centers = np.zeros((len(triangles), 2))
                    for i, tri in enumerate(triangles):
                        tri_centers[i, 0] = np.mean(combined_x[tri])
                        tri_centers[i, 1] = np.mean(combined_y[tri])

                    tri_mask = ~contour_path.contains_points(tri_centers)
                    triang.set_mask(tri_mask)

                    # Get combined min/max for consistent color scaling
                    combined_min = min(np.min(combined_value), np.min(combined_pred))
                    combined_max = max(np.max(combined_value), np.max(combined_pred))

                    # Create subplots with colorbar space
                    # Row 0: Ground Truth
                    ax0 = fig.add_subplot(gs[0, col])
                    im0 = ax0.tripcolor(triang, combined_value, shading='gouraud', cmap=custom_cmap,
                                        vmin=combined_min, vmax=combined_max)
                    ax0.plot(current_contour[:, 0], current_contour[:, 1], 'r-', linewidth=1)

                    # ax0.set_title(f"Slice {slice_idx}")
                    ax0.axis('equal')
                    ax0.axis('off')

                    # Add colorbar to the right of the plot
                    divider0 = make_axes_locatable(ax0)
                    cax0 = divider0.append_axes("right", size="10%" )
                    cbar0 = fig.colorbar(im0, cax=cax0)
                    cbar0.ax.tick_params(labelsize=10 )  # Increase font size
                    cbar0.set_ticks(np.linspace(combined_min, combined_max, 6))

                    fig.colorbar(im0, cax=cax0)

                    # Row 1: Prediction
                    ax1 = fig.add_subplot(gs[1, col])
                    im1 = ax1.tripcolor(triang, combined_pred, shading='gouraud', cmap=custom_cmap,
                                        vmin=combined_min, vmax=combined_max)
                    ax1.plot(current_contour[:, 0], current_contour[:, 1], 'b-', linewidth=1.5)

                    ax1.axis('equal')
                    ax1.axis('off')

                    # Add colorbar to the right of the plot
                    divider1 = make_axes_locatable(ax1)
                    cax1 = divider1.append_axes("right", size="10%" )
                    cbar1 = fig.colorbar(im1, cax=cax1)
                    cbar1.ax.tick_params(labelsize=10 )
                    cbar1.set_ticks(np.linspace(combined_min, combined_max, 6))

                    fig.colorbar(im1, cax=cax1)

                    # Row 2: Absolute Error
                    ax2 = fig.add_subplot(gs[2, col])
                    im2 = ax2.tripcolor(triang, combined_error, shading='gouraud', cmap=custom_cmap)
                    ax2.plot(current_contour[:, 0], current_contour[:, 1], 'b-', linewidth=1.5)
                    if col == 0:
                        ax2.set_ylabel("Absolute Error")
                    ax2.axis('equal')
                    ax2.axis('off')

                    # Add colorbar to the right of the plot
                    divider2 = make_axes_locatable(ax2)
                    cax2 = divider2.append_axes("right", size="10%" )
                    cbar2 = fig.colorbar(im2, cax=cax2)
                    cbar2.ax.tick_params(labelsize=10 )
                    cbar2.set_ticks(np.linspace(np.min(combined_error), np.max(combined_error), 6))

                    fig.colorbar(im2, cax=cax2)

                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, "all_slices_comparison.png"), bbox_inches='tight',dpi = 100,
           transparent=True)
                plt.close()


if __name__ == '__main__':
    save_dir = '.\\save_dir'
    base_log_dir = ('.\\base_log_dir')
    load_path = ".\\whole_model.pth"
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    save_dir = os.path.join(save_dir,time_based_unique_id)
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    val(embed_dim=256, load_path=load_path,output_dir=save_dir)


