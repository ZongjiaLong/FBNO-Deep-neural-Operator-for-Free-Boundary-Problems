import matplotlib.pyplot as plt
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import torch.cuda as cuda
from memory_profiler import memory_usage
import pandas as pd
import numpy as np
from typing import Dict, Optional
import json
import os

from data.G_dataloaders import get_dataloaders
from model.G_model import Model
from utils.loss import setup_seed

_cached_train_loader = None
_cached_val_loader = None
_cached_npz_path = None


def masked_l2_loss(pred, target, mask):
    num_examples = pred.size()[0]
    error = pred - target
    error = error * mask
    target = (target * mask).reshape(num_examples, -1)
    error_norm = torch.norm(error.reshape(num_examples, -1), p=2)
    masked_target = torch.norm(target, p=2)
    l2_error = error_norm / masked_target
    return l2_error
class DataLoaderCache:
    def __init__(self, cache_dir='./dataloader_cache/'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, npz_file_path, batch_size, seed):
        file_mtime = os.path.getmtime(npz_file_path) if os.path.exists(npz_file_path) else 0
        return f"{npz_file_path}_{batch_size}_{seed}_{file_mtime}.pkl"

    def get_dataloaders(self, npz_file_path, batch_size=32, seed=42):
        """
        Get data loaders from cache or create new ones.

        Args:
            npz_file_path: Path to the .npz data file
            batch_size: Batch size for data loaders
            seed: Random seed for reproducibility

        Returns:
            Validation data loader
        """
        cache_file = os.path.join(self.cache_dir, self._get_cache_key(npz_file_path, batch_size, seed))

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError, Exception) as e:
                print(f"Cache loading failed: {e}, creating new data loaders...")

        # Create new data loaders
        print("Creating new data loaders and caching...")
        train_loader, val_loader = get_dataloaders(npz_file_path, batch_size=batch_size, seed=seed)

        # Cache the validation loader
        del train_loader
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((val_loader), f)
            print(f"Data loaders cached to: {cache_file}")
        except Exception as e:
            print(f"Cache saving failed: {e}, but this does not affect usage")

        return val_loader
def plot_error_distributions(m, seed=420,alpha=0.5):
    """
       Plot error distributions for model predictions.

       Args:
           m: Trained model
           seed: Random seed for reproducibility
           alpha: Transparency for the density plot
       """
    setup_seed(seed)

    V_errors = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x_train = batch['x'].to(device)
            y_train = batch['y'].to(device)
            z_train = batch['z'].to(device)
            v_train = batch['v'].to(device)
            mask_train = batch['mask'].to(device)
            t_train = batch['time'].to(device)
            fx_train = batch['fx'].to(device)

            v_pred = m.forward(x=x_train, y=y_train, z=z_train, t=t_train, fx=fx_train)
            loss = masked_l2_loss(v_pred, v_train, mask_train)
            V_errors.extend(loss.cpu().numpy().flatten())

    V_errors = np.array(V_errors)
    v_means = V_errors.mean()
    print(f'V l2 loss mean: {v_means}')
    plt.figure(figsize=(6, 3))
    plt.rcParams['xtick.labelsize'] = 14  # X-axis tick label size
    plt.rcParams['ytick.labelsize'] = 14

    sns.kdeplot(V_errors, fill=True, alpha=alpha, color='lightpink',linewidth=2, label='Density')
    plt.axvline(v_means, color='black', linestyle='--', linewidth=3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    plt.tight_layout()
    plt.savefig('。\\vl2.svg', bbox_inches='tight', transparent=True)
    plt.show()


def measure_model_efficiency(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        num_batches: int = 10,
        num_warmup: int = 2,
        device: Optional[torch.device] = None,
        use_amp: bool = False,
        profile_memory: bool = True
) -> Dict:
    """
    Measure model efficiency, speed, and memory usage.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        num_batches: Number of batches for measurement
        num_warmup: Number of warmup batches
        device: Computation device
        use_amp: Whether to use mixed precision
        profile_memory: Whether to profile memory usage

    Returns:
        Dictionary containing various metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    # Get a sample batch from the data loader
    sample_batch = None
    for batch in val_loader:
        sample_batch = batch
        break

    if sample_batch is None:
        raise ValueError("Data loader is empty")

    # Move sample data to device
    sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample_batch.items()}

    # Warmup phase
    print("Warmup phase...")
    with torch.no_grad():
        for _ in range(num_warmup):
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model.forward(
                        x=sample_batch['x'],
                        y=sample_batch['y'],
                        z=sample_batch['z'],
                        t=sample_batch['time'],
                        fx=sample_batch['fx']
                    )
            else:
                _ = model.forward(
                    x=sample_batch['x'],
                    y=sample_batch['y'],
                    z=sample_batch['z'],
                    t=sample_batch['time'],
                    fx=sample_batch['fx']
                )

    if device.type == 'cuda':
        cuda.synchronize()

    # Measure inference time
    print("Measuring inference time...")
    batch_times = []
    fps_list = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches + num_warmup:
                break
            if i < num_warmup:
                continue

            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Measure time
            start_time = time.perf_counter()

            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model.forward(
                        x=batch['x'],
                        y=batch['y'],
                        z=batch['z'],
                        t=batch['time'],
                        fx=batch['fx']
                    )
            else:
                outputs = model.forward(
                    x=batch['x'],
                    y=batch['y'],
                    z=batch['z'],
                    t=batch['time'],
                    fx=batch['fx']
                )

            if device.type == 'cuda':
                cuda.synchronize()

            end_time = time.perf_counter()
            batch_time = end_time - start_time

            batch_times.append(batch_time)
            batch_size = batch['x'].size(0)
            fps = batch_size / batch_time
            fps_list.append(fps)

    # Measure memory usage
    memory_stats = {}
    if profile_memory:
        print("Measuring memory usage...")

        def inference_func():
            with torch.no_grad():
                batch = sample_batch
                if use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        _ = model.forward(
                            x=batch['x'],
                            y=batch['y'],
                            z=batch['z'],
                            t=batch['time'],
                            fx=batch['fx']
                        )
                else:
                    _ = model.forward(
                        x=batch['x'],
                        y=batch['y'],
                        z=batch['z'],
                        t=batch['time'],
                        fx=batch['fx']
                    )

        # Measure CPU memory
        mem_usage = memory_usage((inference_func, ()), interval=0.1, include_children=True)
        memory_stats['cpu_memory_peak_mb'] = max(mem_usage)
        memory_stats['cpu_memory_avg_mb'] = np.mean(mem_usage)

        # Measure GPU memory
        if device.type == 'cuda':
            cuda.reset_peak_memory_stats(device)

            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model.forward(
                            x=sample_batch['x'],
                            y=sample_batch['y'],
                            z=sample_batch['z'],
                            t=sample_batch['time'],
                            fx=sample_batch['fx']
                        )
                else:
                    _ = model.forward(
                        x=sample_batch['x'],
                        y=sample_batch['y'],
                        z=sample_batch['z'],
                        t=sample_batch['time'],
                        fx=sample_batch['fx']
                    )

            memory_stats['gpu_memory_allocated_mb'] = cuda.memory_allocated(device) / 1024 ** 2
            memory_stats['gpu_memory_cached_mb'] = cuda.memory_reserved(device) / 1024 ** 2
            memory_stats['gpu_memory_peak_mb'] = cuda.max_memory_allocated(device) / 1024 ** 2

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate statistics
    batch_times = np.array(batch_times)
    fps_list = np.array(fps_list)

    metrics = {
        'device': str(device),
        'mixed_precision': use_amp,

        # Time metrics
        'batch_time_mean_ms': np.mean(batch_times) * 1000,
        'batch_time_std_ms': np.std(batch_times) * 1000,
        'batch_time_min_ms': np.min(batch_times) * 1000,
        'batch_time_max_ms': np.max(batch_times) * 1000,

        # Throughput metrics
        'throughput_mean_fps': np.mean(fps_list),
        'throughput_std_fps': np.std(fps_list),
        'throughput_min_fps': np.min(fps_list),
        'throughput_max_fps': np.max(fps_list),

        # Memory metrics
        'memory': memory_stats,

        # Model complexity
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_parameters_m': total_params / 1e6,
        'trainable_parameters_m': trainable_params / 1e6,

        # Efficiency metrics
        'fps_per_million_params': np.mean(fps_list) / (total_params / 1e6) if total_params > 0 else 0,

        # Measurement settings
        'num_batches_measured': len(batch_times),
        'batch_size': sample_batch['x'].size(0) if sample_batch else None,
    }

    return metrics


def print_efficiency_report(metrics: Dict, save_path: Optional[str] = None):
    """Print efficiency report."""
    print("\n" + "=" * 60)
    print("Model Efficiency Evaluation Report")
    print("=" * 60)

    print(f"\n1. Device Information:")
    print(f"   Device: {metrics['device']}")
    print(f"   Mixed Precision: {metrics['mixed_precision']}")

    print(f"\n2. Model Complexity:")
    print(f"   Total Parameters: {metrics['total_parameters_m']:.2f}M")
    print(f"   Trainable Parameters: {metrics['trainable_parameters_m']:.2f}M")

    print(f"\n3. Inference Performance (based on {metrics['num_batches_measured']} batches):")
    print(f"   Average Batch Time: {metrics['batch_time_mean_ms']:.2f} ± {metrics['batch_time_std_ms']:.2f} ms")
    print(f"   Time Range: [{metrics['batch_time_min_ms']:.2f}, {metrics['batch_time_max_ms']:.2f}] ms")
    print(f"   Average Throughput: {metrics['throughput_mean_fps']:.2f} ± {metrics['throughput_std_fps']:.2f} samples/sec")
    print(f"   Throughput Range: [{metrics['throughput_min_fps']:.2f}, {metrics['throughput_max_fps']:.2f}] samples/sec")

    if metrics.get('memory'):
        print(f"\n4. Memory Usage:")
        if 'cpu_memory_peak_mb' in metrics['memory']:
            print(f"   CPU Memory Peak: {metrics['memory']['cpu_memory_peak_mb']:.2f} MB")
            print(f"   CPU Memory Average: {metrics['memory']['cpu_memory_avg_mb']:.2f} MB")

        if 'gpu_memory_peak_mb' in metrics['memory']:
            print(f"   GPU Memory Peak: {metrics['memory']['gpu_memory_peak_mb']:.2f} MB")
            print(f"   GPU Memory Allocated: {metrics['memory']['gpu_memory_allocated_mb']:.2f} MB")
            print(f"   GPU Memory Cached: {metrics['memory']['gpu_memory_cached_mb']:.2f} MB")

    print(f"\n5. Efficiency Metrics:")
    print(f"   Throughput per Million Parameters: {metrics['fps_per_million_params']:.2f} samples/sec/M")

    print("\n" + "=" * 60)

    # Save to file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Convert to serializable format
        report_data = metrics.copy()
        report_data['device'] = str(report_data['device'])

        with open(save_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nReport saved to: {save_path}")

        # Also save in CSV format
        csv_path = save_path.replace('.json', '.csv')
        flat_data = {}
        for key, value in report_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_data[f"{key}_{sub_key}"] = sub_value
            else:
                flat_data[key] = value

        df = pd.DataFrame([flat_data])
        df.to_csv(csv_path, index=False)
        print(f"CSV report saved to: {csv_path}")


def compare_efficiency_metrics(
        models: Dict[str, torch.nn.Module],
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_batches: int = 10,
        save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare efficiency metrics of multiple models.

    Args:
        models: Dictionary of models {model_name: model_instance}
        val_loader: Data loader
        device: Computation device
        num_batches: Number of batches for measurement
        save_path: Save path for results

    Returns:
        DataFrame with comparison results
    """
    results = []

    for name, model in models.items():
        print(f"\nEvaluating model: {name}")
        print("-" * 40)

        # Evaluate different configurations
        for use_amp in [False, True] if device.type == 'cuda' else [False]:
            config_name = f"{name}_amp" if use_amp else name

            try:
                metrics = measure_model_efficiency(
                    model=model,
                    val_loader=val_loader,
                    num_batches=num_batches,
                    device=device,
                    use_amp=use_amp
                )

                result = {
                    'model_name': name,
                    'mixed_precision': use_amp,
                    'throughput_fps': metrics['throughput_mean_fps'],
                    'batch_time_ms': metrics['batch_time_mean_ms'],
                    'memory_peak_mb': metrics['memory'].get('gpu_memory_peak_mb',
                                                            metrics['memory'].get('cpu_memory_peak_mb', 0)),
                    'total_params_m': metrics['total_parameters_m'],
                    'fps_per_million_params': metrics['fps_per_million_params']
                }
                results.append(result)

                print(f"  Configuration: {config_name}")
                print(f"  Throughput: {metrics['throughput_mean_fps']:.2f} samples/sec")
                print(f"  Batch Time: {metrics['batch_time_mean_ms']:.2f} ms")

            except Exception as e:
                print(f"  Evaluation failed: {e}")

    # Create comparison table
    df = pd.DataFrame(results)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nComparison results saved to: {save_path}")

    return df


if __name__ == '__main__':
    npz_file_path = r"D:\download\all_data_15000.npz"
    seed = 42
    cache = DataLoaderCache()
    val_loader = cache.get_dataloaders(npz_file_path, batch_size=24, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_path = "D:\download\G_model.pth"

    # setup_seed(seed)
    # m = Model(embed_dim=128, device=device)
    # m.load_state_dict(torch.load(load_path))
    # m = m.to(device)
    # m.eval()
    # plot_error_distributions(m,seed=seed)
    setup_seed(seed)
    m = Model(embed_dim=128, device=device)
    m.load_state_dict(torch.load(load_path))
    m = m.to(device)
    m.eval()

    # 1. Single model evaluation
    print("Single model efficiency evaluation...")
    metrics = measure_model_efficiency(
        model=m,
        val_loader=val_loader,
        num_batches=20,
        num_warmup=3,
        device=device,
        use_amp=True,  # Try mixed precision
        profile_memory=True
    )

    # Print detailed report
    print_efficiency_report(
        metrics,
        save_path="./efficiency_report.json"  # Optional: save report
    )