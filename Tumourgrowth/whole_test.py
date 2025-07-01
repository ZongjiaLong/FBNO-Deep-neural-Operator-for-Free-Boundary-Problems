from scipy.interpolate import make_interp_spline
import os
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import time
from data.value_data import create_dataloaders
from model.Whole_model import Model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




myloss = LpLoss(size_average=False)
MSE_loss = torch.nn.HuberLoss()
l2_loss = LpLoss(size_average=False)


def vis_vio(embed_dim=64, seed=420, load_path=None, custom_colors=None,size= (10,6)):
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
    print(f"Number of trainable parameters: {a}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize metrics and error collection
    test_sup = 0
    total_samples = 0
    batch_times = []

    # To store errors and domain information
    error_data = []
    domain_data = []
    unique_domains = set()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_start_time = time.time()

            value_test = batch.value.to(device)
            coord_test = batch.coord.to(device)
            fx_test = batch.fx.to(device)
            initial_domain_test = batch.initial_domain.to(device)
            x_test, y_test, t_test = coord_test[:, :, 0], coord_test[:, :, 1], coord_test[:, :, 2]

            u_hat_test = m.forward(x=x_test, y=y_test, t=t_test,
                                   init_domain=initial_domain_test, fx=fx_test)

            # Calculate individual errors for each sample in the batch
            batch_errors = torch.abs((u_hat_test.squeeze() - value_test)).cpu().numpy()

            # batch_errors = torch.sqrt((u_hat_test.squeeze() - value_test)**2).cpu().numpy()
            initial_domain_test = initial_domain_test.squeeze()

            # Convert initial_domain_test to domain identifiers
            # Collect unique domains and create domain labels
            domain_labels = []
            for domain in initial_domain_test:
                # Convert domain tensor to a hashable tuple
                domain_tuple = tuple(domain.cpu().numpy().tolist())
                unique_domains.add(domain_tuple)
                domain_labels.append(domain_tuple)

            # Flatten the arrays for processing
            flat_errors = batch_errors.flatten()

            # Store errors and corresponding domain types
            for error, domain in zip(flat_errors, domain_labels):
                domain_data.append(str(domain))
                error_data.append(error)# Convert to string for labeling

            loss_sup_test = l2_loss(u_hat_test, value_test)
            test_sup += loss_sup_test.item() * len(value_test)
            total_samples += len(value_test)

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

    # Create mapping from domain tuples to alphabetical labels
    domain_mapping = {domain: chr(65 + i) for i, domain in enumerate(unique_domains)}

    # Create DataFrame for visualization with alphabetical labels
    error_df = pd.DataFrame({
        'Error': error_data,
        'Domain Type': [domain_mapping[tuple(eval(d))] for d in domain_data]
    })

    # Apply log transformation to the error data
    error_df['Log_Error'] = np.log(error_df['Error'])
    if custom_colors is None:
        custom_colors = ["#00FF7F", "#FF7F50", "#FFD700"]  # Example default colors

    # Plotting with minimal styling
    plt.figure(figsize=size)

    # Set transparent background
    plt.gcf().patch.set_alpha(0)

    # Create the violin plot
    ax = sns.violinplot(x='Domain Type', y='Log_Error', data=error_df,
                        palette=custom_colors,
                        cut=0,
                        alpha=0.5)

    # Remove all borders and ticks except x and y axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)  # Keep x-axis
    ax.spines['left'].set_visible(True)  # Keep y-axis

    # Remove all tick marks
    ax.tick_params(axis='both', which='both', length=0)

    # Remove x-axis labels and ticks
    ax.set_xticklabels([])
    ax.set_xlabel('')

    # Keep y-axis labels but make them larger
    ax.tick_params(axis='y', labelsize=28)  # Larger font for y-axis ticks

    # Set transparent background for the plot area
    ax.patch.set_alpha(0)

    plt.tight_layout()

    # Second plot with anti-log y-axis ticks
    plt.figure(figsize=(max(10, len(unique_domains)), 6))

    # Set transparent background
    plt.gcf().patch.set_alpha(0)

    ax = sns.violinplot(x='Domain Type', y='Log_Error', data=error_df,
                        palette=custom_colors,
                        cut=0,
                        alpha=0.5)

    # Remove all borders and ticks except x and y axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Remove all tick marks
    ax.tick_params(axis='both', which='both', length=0)

    # Remove x-axis labels and ticks
    ax.set_xticklabels([])
    ax.set_xlabel('')

    # Set y-axis with anti-log ticks and larger font
    y_ticks = ax.get_yticks()
    # ax.set_yscale('log')

    ax.set_yticklabels([f"{np.exp(y):.2g}" for y in y_ticks], fontsize=28)
    ax.set_ylabel('', fontsize=24)

    # Set transparent background for the plot area
    ax.patch.set_alpha(0)

    plt.tight_layout()
    print("\nAverage Errors by Domain Type:")
    for domain_label in sorted(error_df['Domain Type'].unique()):
        avg_error = error_df[error_df['Domain Type'] == domain_label]['Error'].mean()
        print(f"Domain {domain_label}: {avg_error:.4f}")

    # Calculate and print overall average error
    overall_avg = error_df['Error'].mean()
    print(f"\nOverall Average Error: {overall_avg:.4f}")
    output_path = '.\\result'
    plot_filename = os.path.join(output_path, 'Tumourvio.svg')
    plt.savefig(plot_filename,  bbox_inches='tight')
    plt.show()

def vis_time_dis(embed_dim=64, seed=420, load_path=None,size = (10,6)):
    setup_seed(seed)
    m = Model(embed_dim=embed_dim, pretrain_path='.\\boundary_mode_three.pth')

    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(
                f"Error loading state_dict: {e}. Check model architecture or compatibility. Training from scratch.")

    m = m.to('cuda')
    a = count_params_with_grad(m)
    print(f"Number of trainable parameters: {a}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    error_dict = {}  # Key: time, Value: list of errors at that time
    count_dict = {}  # Key: time, Value: sample count at that time

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            value_test = batch.value.to(device)
            coord_test = batch.coord.to(device)
            fx_test = batch.fx.to(device)
            initial_domain_test = batch.initial_domain.to(device)
            x_test, y_test, t_test = coord_test[:, :, 0], coord_test[:, :, 1], coord_test[:, :, 2]

            u_hat_test = m.forward(x=x_test, y=y_test, t=t_test,
                                   init_domain=initial_domain_test, fx=fx_test)

            # Calculate per-sample errors and times
            batch_errors = torch.abs((u_hat_test.squeeze() - value_test)).cpu().numpy()

            # batch_errors = torch.sqrt((u_hat_test.squeeze() - value_test)**2).cpu().numpy()
            batch_times_np = (t_test*150).cpu().numpy()  # Get all time values

            # Store errors and counts by their corresponding time points
            for t, err in zip(batch_times_np.flatten(), batch_errors.flatten()):
                if t not in error_dict:
                    error_dict[t] = []
                    count_dict[t] = 0
                error_dict[t].append(err)
                count_dict[t] += 1

    # Prepare data for plotting
    times = sorted(error_dict.keys())
    mean_errors = []
    std_errors = []
    counts = []

    for t in times:
        errors_at_t = error_dict[t]
        if len(errors_at_t) > 0:
            upper_bound = np.percentile(errors_at_t, 95)

            # Filter errors within the 95% range
            filtered_errors = [err for err in errors_at_t if err <= upper_bound]

            if filtered_errors:  # Only add if there are values in the filtered range
                mean_errors.append(np.mean(filtered_errors))
                std_errors.append(np.std(filtered_errors))
                counts.append(len(filtered_errors))
            else:
                # If no values in filtered range, skip this time point
                continue
        else:
            # If no errors at this time, skip
            continue

    # Create smooth curves using spline interpolation
    times_np = np.array(times)
    mean_errors_np = np.array(mean_errors)
    std_errors_np = np.array(std_errors)

    # Generate dense time points for smooth curves
    dense_times = np.linspace(min(times), max(times), 500)
    # Create spline for mean errors
    spl_mean = make_interp_spline(times_np, mean_errors_np, k=3)
    smooth_mean = spl_mean(dense_times)

    # Create spline for standard deviations
    spl_std = make_interp_spline(times_np, std_errors_np, k=3)
    smooth_std = spl_std(dense_times)
    a = smooth_mean - smooth_std
    # Visualization
    plt.rcParams.update({
        'figure.facecolor': 'none',  # Figure background transparent
        'axes.facecolor': 'none',  # Axes background transparent
        'savefig.facecolor': 'none',  # Save figure with transparent background
        'xtick.labelsize': 28,  # x-axis tick label size
        'ytick.labelsize': 28  # y-axis tick label size
    })
    plt.figure(figsize=size)

    # Create a twin axis for the count bars
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot light green bars for sample counts (no label)
    ax2.bar(times, counts, width=5, color='lightgreen', alpha=0.7)
    # Remove right y-axis labels and ticks
    ax2.set_yticklabels([])
    ax2.set_yticks([])

    # Remove top spine from both axes
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.spines['right'].set_visible(False)

    # Plot uncertainty band (mean ± std) on primary axis (no label)

    ax1.fill_between(dense_times,
                     smooth_mean - 1.96*smooth_std,
                     smooth_mean + 1.96*smooth_std,
                     color='skyblue', alpha=0.4)

    # Plot smooth mean error curve (no label)
    ax1.plot(dense_times, smooth_mean, color='royalblue', linewidth=2.5)

    # Plot original points
    ax1.scatter(times, mean_errors, color='navy', s=40, edgecolor='white')
    # ax1.set_yscale('log')

    # Remove grid and legend
    ax1.grid(False)
    plt.tight_layout()
    # Calculate and print overall mean error

    output_path = '.\\result'
    plot_filename = os.path.join(output_path, 'Tumourtime.svg')
    plt.savefig(plot_filename,  bbox_inches='tight')
    plt.show()

    return times, mean_errors, std_errors, counts


def vis_dis(embed_dim=64, seed=420, load_path=None,size = (10,6)):
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
    print(f"Number of trainable parameters: {a}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize metrics and storage for per-sample errors
    l2_errors = []
    abs_errors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            value_test = batch.value.to(device)
            coord_test = batch.coord.to(device)
            fx_test = batch.fx.to(device)
            initial_domain_test = batch.initial_domain.to(device)
            x_test, y_test, t_test = coord_test[:, :, 0], coord_test[:, :, 1], coord_test[:, :, 2]

            u_hat_test = m.forward(x=x_test, y=y_test, t=t_test,
                                   init_domain=initial_domain_test, fx=fx_test)

            # Calculate per-sample errors
            batch_l2_errors = torch.norm(u_hat_test.squeeze() - value_test, p=2, dim=1) / torch.norm(value_test, p=2,
                                                                                                     dim=1)
            batch_abs_errors = torch.mean((u_hat_test.squeeze() - value_test)**2, dim=1)

            # Store errors for visualization/analysis
            l2_errors.extend(batch_l2_errors.cpu().numpy())
            abs_errors.extend(batch_abs_errors.cpu().numpy())

    # Convert to numpy arrays for analysis
    l2_errors = np.array(l2_errors)
    abs_errors = np.array(abs_errors)

    # Calculate mean errors
    mean_l2 = np.mean(l2_errors)
    mean_abs = np.mean(abs_errors)
    print(f"Mean L2 error: {mean_l2}")
    print(f"Mean Absolute error: {mean_abs}")
    # Plot error distributions using KDE
    plt.figure(figsize=size, facecolor='none')  # Taller figure for vertical layout and transparent background
    plt.rcParams['xtick.labelsize'] = 28 # X-axis tick label size

    # Make the figure background transparent
    plt.gcf().set_facecolor('none')

    # L2 Error plot (top)
    plt.subplot(2, 1, 1)
    sns.kdeplot(l2_errors, color='blue', shade=True, alpha=0.3, log_scale=True)
    plt.axvline(mean_l2, color='black', linestyle='--', linewidth=5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)  # Keep bottom spine for reference
    plt.yticks([])
    plt.ylabel('')

    # Absolute Error plot (bottom)
    plt.subplot(2, 1, 2)
    sns.kdeplot(abs_errors, color='green', shade=True, alpha=0.3, log_scale=True)
    plt.axvline(mean_abs, color='black', linestyle='--', linewidth=5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)  # Keep bottom spine for reference
    plt.yticks([])
    plt.ylabel('')

    plt.tight_layout()

    output_path = '.\\result'
    plot_filename = os.path.join(output_path, 'Tumourvis.svg')
    plt.savefig(plot_filename, bbox_inches='tight', transparent=True,)
    plt.show()



def onlytest(embed_dim=64, seed=420, load_path=None):
    setup_seed(seed)
    m = Model(embed_dim=embed_dim, pretrain_path='.\\boundary_mode_three.pth')

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
    print(f"Number of trainable parameters: {a}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize metrics
    test_sup = 0
    total_samples = 0
    batch_times = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Start timer for this batch
            batch_start_time = time.time()

            value_test = batch.value.to(device)
            coord_test = batch.coord.to(device)
            fx_test = batch.fx.to(device)
            initial_domain_test = batch.initial_domain.to(device)
            x_test, y_test, t_test = coord_test[:, :, 0], coord_test[:, :, 1], coord_test[:, :, 2]

            u_hat_test = m.forward(x=x_test, y=y_test, t=t_test,
                                   init_domain=initial_domain_test, fx=fx_test)

            loss_sup_test = l2_loss(u_hat_test, value_test)
            test_sup += loss_sup_test.item() * len(value_test)  # Weight by batch size
            total_samples += len(value_test)

            # End timer and record
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                avg_time = sum(batch_times) / len(batch_times)
                remaining_batches = len(test_loader) - (batch_idx + 1)
                eta = remaining_batches * avg_time
                print(f"Batch {batch_idx + 1}/{len(test_loader)} | "
                      f"Avg batch time: {avg_time:.4f}s | "
                      f"ETA: {eta:.1f}s")

    # Calculate final metrics
    sup = test_sup / total_samples  # Average over all samples
    avg_batch_time = sum(batch_times) / total_samples
    total_time = sum(batch_times)

    print("\nFinal Results:")
    print(f"Average sample processing time: {avg_batch_time:.4f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average L2 error per sample: {sup:.6f}")
    print(f"Processed {total_samples} samples in {len(test_loader)} batches")

    return sup



if __name__ == '__main__':
    test_path = ".\\domain_test.npy"

    test_loader = create_dataloaders(test_path, batch_size=60, shuffle=False, max_samples=900)
    load_path = ".\\whole_model.pth"
    # vis_time_dis(embed_dim=256,load_path=load_path)
    # vis_vio(embed_dim=256,load_path=load_path)
    vis_dis(embed_dim=256,load_path=load_path,size=(10,4))
