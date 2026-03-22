import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import random
from model.H_model import TS_H_model
import matplotlib.pyplot as plt
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import matplotlib

import optuna
import time


def get_time(val_loader, model_path, target_batch_idx=63, target_sample_idx=16, batchsize=200,
             n_iterations=int(1e3)):
    """
    Measure the inference time of the model by performing random sampling.

    Args:
        val_loader: Validation data loader
        model_path: Path to the trained model
        target_batch_idx: Index of the target batch
        target_sample_idx: Index of the target sample within the batch
        batchsize: Batch size for inference
        n_iterations: Number of iterations to run for timing

    Returns:
        Tuple of (best_fx, timing_info)
    """
    setup_seed(58)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TS_H_model(num_layers=6, embed_dim=640)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Find target batch
    target_batch = None
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx == target_batch_idx:
            target_batch = batch
            break

    # Extract target data
    x_target = target_batch['x'][target_sample_idx].to(device)
    phi_target = target_batch['phi'][target_sample_idx].to(device)
    fx_target = target_batch['fx'][target_sample_idx].to(device)

    print(f'Target t: {fx_target[2].item():.8f}')
    print(f'Target fx: [{fx_target[0].item():.8f}, {fx_target[1].item():.8f}]')

    # Forward pass with target condition
    with torch.no_grad():
        pred_phi, x_out = model.forward(
            x=x_target.unsqueeze(0),
            condition=fx_target.unsqueeze(0)
        )

    # Calculate phi error
    phi_error = torch.mean(torch.abs(pred_phi.squeeze() - phi_target))
    print(f'Target phi error: {phi_error:.8f}')

    # Initialize tracking variables
    best_loss = float('inf')
    best_fx = None
    phi_target = phi_target.unsqueeze(0).expand(batchsize, -1, -1)
    x_batch = x_target.unsqueeze(0).expand(batchsize, -1, -1)

    # Timing variables
    total_time = 0.0
    total_batches = 0
    total_samples_generated = 0

    # Main timing loop
    pbar = tqdm(range(n_iterations), desc="Randomly searching")
    for iteration in pbar:
        # Generate random fx values
        fx_random = torch.rand(batchsize, 3, device=device)  # shape: (batchsize, 3)

        # Start timing
        start_time = time.time()

        # Forward pass
        with torch.no_grad():
            pred_phi, x_out = model.forward(
                x=x_batch,
                condition=fx_random
            )

        # End timing
        batch_time = time.time() - start_time

        # Update timing statistics
        total_time += batch_time
        total_batches += 1
        total_samples_generated += batchsize

        # Calculate errors
        phi_errors = torch.mean(torch.abs(pred_phi.reshape(batchsize, -1) - phi_target.reshape(batchsize, -1)),
                                dim=1)  # shape: (batchsize,)

        # Update progress bar with timing info
        avg_time_per_sample = total_time / total_samples_generated
        avg_time_per_batch = total_time / total_batches
        pbar.set_postfix({
            'avg_sample_time': f'{avg_time_per_sample:.6f}s',
            'avg_batch_time': f'{avg_time_per_batch:.3f}s',
        })

    # Print final timing statistics
    print("\n" + "=" * 50)
    print("Timing Statistics:")
    print("=" * 50)
    print(f"Total iterations: {n_iterations}")
    print(f"Total batches: {total_batches}")
    print(f"Total samples generated: {total_samples_generated}")
    print(f"Total runtime: {total_time:.3f} seconds")
    print(f"Average time per batch: {total_time / total_batches:.3f} seconds")
    print(f"Average time per sample: {total_time / total_samples_generated:.6f} seconds")
    print(f"Sample generation speed: {total_samples_generated / total_time:.1f} samples/second")

    # GPU memory info if available
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

    return best_fx, {
        'total_time': total_time,
        'avg_sample_time': total_time / total_samples_generated,
        'avg_batch_time': total_time / total_batches,
        'samples_per_second': total_samples_generated / total_time
    }


def search_approximation(val_loader, model_path, target_batch_idx=101,
                         target_sample_idx=11, batchsize=200, n_trials=160):
    """
    Use Bayesian optimization (Optuna) to search for fx values that minimize the prediction error.

    Args:
        val_loader: Validation data loader
        model_path: Path to the trained model
        target_batch_idx: Index of the target batch
        target_sample_idx: Index of the target sample within the batch
        batchsize: Batch size for evaluation
        n_trials: Number of optimization trials

    Returns:
        Tuple of (best_fx, best_error, study)
    """
    setup_seed(58)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TS_H_model(num_layers=6, embed_dim=640)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Find target batch
    target_batch = None
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx == target_batch_idx:
            target_batch = batch
            break

    # Extract target data
    x_target = target_batch['x'][target_sample_idx].to(device)
    phi_target = target_batch['phi'][target_sample_idx].to(device)
    fx_target = target_batch['fx'][target_sample_idx].to(device)

    # Forward pass with target condition
    with torch.no_grad():
        pred_phi, x_out = model.forward(
            x=x_target.unsqueeze(0),
            condition=fx_target.unsqueeze(0)
        )

    # Calculate baseline error
    ori_error = torch.mean(torch.abs(pred_phi.squeeze() - phi_target)).item()
    print(f'Target t: {fx_target[2].item():.8f}')
    print(f'Target fx: [{fx_target[0].item():.8f}, {fx_target[1].item():.8f}]')
    print(f'Target phi error: {ori_error:.8f}')

    def objective(trial):
        """Objective function for Bayesian optimization, optimizing a single fx sample."""
        # Suggest fx values
        fx0 = trial.suggest_float('fx0', -5, 5)
        fx1 = trial.suggest_float('fx1', -5, 5)
        fx2 = trial.suggest_float('fx2', 0.0, 1.0)

        # Create batch with noise around suggested values
        batchsize = 50
        fx_random = torch.randn(batchsize, 3, device=device) * 0.05
        fx_random[:, 0] += fx0
        fx_random[:, 1] += fx1
        fx_random[:, 2] += fx2

        x_batch = x_target.unsqueeze(0).expand(batchsize, -1, -1)
        phi_batch = phi_target.unsqueeze(0).expand(batchsize, -1, -1)

        # Forward pass
        with torch.no_grad():
            pred_phi, x_out = model.forward(
                x=x_batch,
                condition=fx_random
            )

        # Calculate errors
        errors = torch.mean(torch.abs(pred_phi.reshape(batchsize, -1) -
                                      phi_batch.reshape(batchsize, -1)), dim=1)
        min_error, min_idx = torch.min(errors, dim=0)

        return min_error

    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=88, n_startup_trials=20),
    )

    # Run optimization
    print("Starting Bayesian optimization search...")
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials)

    # Initialize best results
    best_error = study.best_value
    best_params = study.best_params
    best_fx = torch.tensor([best_params['fx0'], best_params['fx1'], best_params['fx2']])

    # Local refinement around best point
    batchsize = 200
    for iteration in tqdm(range(10)):
        # Random sampling around best point
        fx_random = torch.randn(batchsize, 3, device=device) * 0.05
        fx_random[:, 0] += best_params['fx0']
        fx_random[:, 1] += best_params['fx1']
        fx_random[:, 2] += best_params['fx2']

        x_batch = x_target.unsqueeze(0).expand(batchsize, -1, -1)
        phi_batch = phi_target.unsqueeze(0).expand(batchsize, -1, -1)

        with torch.no_grad():
            pred_phi, x_out = model.forward(
                x=x_batch,
                condition=fx_random
            )

        errors = torch.mean(torch.abs(pred_phi.reshape(batchsize, -1) -
                                      phi_batch.reshape(batchsize, -1)), dim=1)
        min_error, min_idx = torch.min(errors, dim=0)

        if min_error < best_error:
            improvement = ((best_error - min_error) / best_error) * 100
            print(f'Found better parameters, improved by {improvement:.2f}%')
            best_error = min_error.item()
            best_fx = fx_random[min_idx].cpu().numpy()

    # Print results summary
    print("\n" + "=" * 50)
    print("Optimization Results Summary:")
    batch_time = time.time() - start_time
    print(f'Time taken: {batch_time:.3f} seconds')
    print("=" * 50)
    print(f"Target fx: [{fx_target[0].item():.8f}, {fx_target[1].item():.8f}, {fx_target[2].item():.8f}]")
    print(f"Best found fx: [{best_fx[0]:.8f}, {best_fx[1]:.8f}, {best_fx[2]:.8f}]")
    print(f"Minimum error: {best_error:.8f}")
    print(f'Original error: {ori_error:.8f}')

    # Calculate relative errors
    target_array = fx_target.cpu().numpy()
    if isinstance(best_fx, torch.Tensor):
        best_array = best_fx.numpy()
    else:
        best_array = best_fx

    abs_diff = np.abs(target_array - best_array)
    rel_diff = abs_diff / (np.abs(target_array) + 1e-8)

    print(f"Absolute error: [{abs_diff[0]:.8f}, {abs_diff[1]:.8f}, {abs_diff[2]:.8f}]")
    print(f"Relative error: [{rel_diff[0]:.4%}, {rel_diff[1]:.4%}, {rel_diff[2]:.4%}]")
    print("=" * 50)

    # Print optimization statistics
    print("\nOptimization Statistics:")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best parameters: {study.best_params}")
    print(f"Best trial number: {study.best_trial.number}")

    return best_fx, best_error, study

def plot_error_distributions(
        model_path,
        data_loader,
        device=None,
        save_path=True,
        figsize=(6, 3),
        alpha=0.7
):
    """
        Plot error distributions for phi and x predictions.

        Args:
            model_path: Path to the trained model
            data_loader: Data loader for evaluation
            device: Device to run on
            save_path: Whether to save the plots
            figsize: Figure size
            alpha: Transparency for plots
        """
    setup_seed(58)
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Font that supports math symbols
    matplotlib.rcParams['mathtext.default'] = 'regular'  # Use regular math font
    matplotlib.rcParams['axes.unicode_minus'] = True

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TS_H_model(num_layers=6, embed_dim=640)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    phi_errors = []
    x_errors = []

    # Calculate errors
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating errors"):
            x = batch['x'].to(device)
            phi = batch['phi'].to(device)
            fx = batch['fx'].to(device)
            shape = fx.shape[0]

            # Forward pass
            pred_phi, x_out = model(x=x, condition=fx)

            phi_error = torch.mean(torch.abs(pred_phi.reshape(shape, -1) - phi.reshape(shape, -1)), dim=-1)
            phi_errors.extend(phi_error.cpu().numpy().flatten())

            x_error = torch.mean(torch.abs(x_out.reshape(shape, -1) - x.reshape(shape, -1)), dim=-1)
            x_errors.extend(x_error.cpu().numpy().flatten())

    # Convert to numpy arrays
    phi_errors = np.array(phi_errors)
    x_errors = np.array(x_errors)

    # Calculate mean errors
    phi_mean = phi_errors.mean()
    x_mean = x_errors.mean()
    print(f'Phi loss mean: {phi_mean}')
    print(f'X loss mean: {x_mean}')

    # Set font sizes
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    # 1. Plot phi errors distribution
    plt.figure(figsize=figsize)
    sns.kdeplot(phi_errors, fill=True, alpha=alpha, color='lightsalmon',
                linewidth=2, log_scale=True, label='Density')

    # Add mean line
    plt.axvline(phi_mean, color='black', linestyle='--', linewidth=3,
                label=f'Mean: {phi_mean:.4e}')
    plt.xlim(2e-3, 1e-1)

    # Remove spines for cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    plt.tight_layout()

    # Save first plot
    if save_path:
        plt.savefig('.\\phi1.svg', bbox_inches='tight', transparent=True)

    plt.show()

    # 2. Plot x errors distribution
    plt.figure(figsize=figsize)
    sns.kdeplot(x_errors, fill=True, alpha=alpha, color='lightgreen',
                linewidth=2, log_scale=True, label='Density')

    plt.axvline(x_mean, color='black', linestyle='--', linewidth=3,
                label=f'Mean: {x_mean:.4e}')

    plt.xlim(2e-3, 1e-1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    plt.tight_layout()

    # Save second plot
    if save_path:
        plt.savefig('.\\aoe1.svg', bbox_inches='tight', transparent=True)

    plt.show()

    return phi_errors, x_errors


def visualize_comparison_grid(model_path, data_loader, num_samples=4,
                              save_dir='D:\desktop\output861\PHI_compare', device=None):
    """
    Visualize comparison between true and predicted phi values for multiple samples.

    Args:
        model_path: Path to the trained model
        data_loader: Data loader for samples
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
        device: Device to run on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_seed(58)

    # Load model
    m = TS_H_model(num_layers=6, embed_dim=512 + 128)

    try:
        m.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    m = m.to(device)
    m.eval()

    # Get data
    batch = next(iter(data_loader))
    x = batch['x'].to(device)
    phi = batch['phi'].to(device)
    fx = batch['fx'].to(device)

    batch_size = x.shape[0]
    if num_samples > batch_size:
        num_samples = batch_size

    # Randomly select samples
    indices = random.sample(range(batch_size), num_samples)

    # Forward pass
    with torch.no_grad():
        pred_phi, x_out = m(x=x, condition=fx)

    x_np = x.cpu().numpy()
    phi_np = phi.cpu().numpy()
    pred_phi_np = pred_phi.cpu().numpy()

    # Create save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"PDF files will be saved to: {save_dir}")

    # Set font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 10,
    })

    # Create separate plots for each sample
    for i, idx in enumerate(indices):
        phi_true = phi_np[idx]
        phi_pred = pred_phi_np[idx]

        # Extract x and y components
        phi_true_x = phi_true[:, 0]
        phi_true_y = phi_true[:, 1]
        phi_pred_x = phi_pred[:, 0]
        phi_pred_y = phi_pred[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Determine plot limits
        all_x = np.concatenate([phi_true_x, phi_pred_x])
        all_y = np.concatenate([phi_true_y, phi_pred_y])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        # Plot true phi
        ax1 = axes[0]
        scatter1 = ax1.scatter(phi_true_x, phi_true_y, c='blue', s=30, alpha=0.7,
                               edgecolors='darkblue', linewidths=0.8)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_aspect('equal', adjustable='box')

        # Plot predicted phi
        ax2 = axes[1]
        scatter2 = ax2.scatter(phi_pred_x, phi_pred_y, c='red', s=30, alpha=0.7,
                               edgecolors='darkred', linewidths=0.8)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_aspect('equal', adjustable='box')

        # Save as PDF
        if save_dir:
            pdf_path = os.path.join(save_dir, f'phi_comparison_sample_{i + 1:03d}_idx_{idx:04d}.svg')
            plt.savefig(pdf_path, bbox_inches='tight', format='svg')
            print(f"Saved: {pdf_path}")

        plt.close(fig)

# Main execution block
if __name__ == '__main__':
    # Set random seed
    seed = 58

    # Define data paths
    output_file_path = r".\TS_data_normalized.npz"
    uv_path = r".\TS_A_coord.npy"
    batchsize = 100

    # Create data loaders
    from data.Get_dataloader import create_data_loaders

    train_loader, val_loader = create_data_loaders(output_file_path, uv_path, batch_size=batchsize, random_seed=seed)

    print(f"Train loader size: {len(train_loader)} batches")
    print(f"Validation loader size: {len(val_loader)} batches")
    print(f'Total dataset size: {len(train_loader) * batchsize} samples')

    # Define model and save paths
    model_path = ".\TSHmodel.pth"  # Trained model path
    save_viz_dir = ".\PHI_compare"  # Visualization save directory

    # Uncomment the functions you want to run:

    # 1. Visualize predictions
    # visualize_predictions(
    #     model_path=model_path,
    #     data_loader=val_loader,  # Using validation set
    #     num_samples=5,  # Visualize 5 samples
    #     save_dir=save_viz_dir
    # )

    # 2. Visualize comparison grid
    # visualize_comparison_grid(
    #     model_path=model_path,
    #     data_loader=val_loader,
    #     num_samples=30,  # 4 samples in 2x2 grid
    #     save_dir=save_viz_dir
    # )

    # 3. Random search for approximation
    # random_search_v_approximation(model_path=model_path, val_loader=val_loader)

    # 4. Bayesian optimization search
    # search_approximation(model_path=model_path, val_loader=val_loader)

    # 5. Timing analysis
    get_time(model_path=model_path, val_loader=val_loader)

    # 6. Plot error distributions
    # plot_error_distributions(
    #     model_path=model_path,
    #     data_loader=val_loader,)
