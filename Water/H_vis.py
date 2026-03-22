import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.phi_iso_dataloaders import create_data_loaders
from model.H_model import Model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import matplotlib
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np


def masked_l2_loss(pred, target, mask):
    """
        Calculate masked L2 loss between prediction and target.

        Args:
            pred: predicted values
            target: ground truth values
            mask: binary mask indicating valid regions

        Returns:
            L2 loss normalized by masked target norm
        """
    num_examples = pred.size()[0]
    error = pred.reshape(num_examples, -1) - target.reshape(num_examples, -1)
    target = target.reshape(num_examples, -1)
    error_norm = torch.norm(error * mask.float(), p=2)
    masked_target= torch.norm(target * mask.float(), p=2)
    l2_error = torch.sum(error_norm/masked_target)
    return l2_error
def masked_l1_loss(pred, target, mask):
    """
        Calculate masked L1 loss between prediction and target.

        Args:
            pred: predicted values
            target: ground truth values
            mask: binary mask indicating valid regions

        Returns:
            L1 loss normalized by masked target norm
        """
    num_examples = pred.size()[0]
    error = pred.reshape(num_examples, -1) - target.reshape(num_examples, -1)
    target = target.reshape(num_examples, -1)
    error_norm = torch.norm(error * mask.float(), p=1)
    masked_target= torch.norm(target * mask.float(), p=1)
    l2_error = torch.sum(error_norm/masked_target)
    return l2_error
def masked_l2_loss_re(pred, target, mask):
    """
        Calculate masked L2 loss (relative) between prediction and target.

        Args:
            pred: predicted values
            target: ground truth values
            mask: binary mask indicating valid regions

        Returns:
            Relative L2 loss (not summed)
        """
    num_examples = pred.size()[0]
    error = pred.reshape(num_examples, -1) - target.reshape(num_examples, -1)
    target = target.reshape(num_examples, -1)
    error_norm = torch.norm(error * mask.float(), p=2)
    masked_target= torch.norm(target * mask.float(), p=2)
    l2_error = error_norm/masked_target
    return l2_error
def masked_l1(pred, target, mask):
    """
        Calculate masked L1 error (relative) between prediction and target.

        Args:
            pred: predicted values
            target: ground truth values
            mask: binary mask indicating valid regions

        Returns:
            Relative L1 error
        """
    num_examples = pred.size()[0]
    error = pred.reshape(num_examples, -1) - target.reshape(num_examples, -1)
    target = target.reshape(num_examples, -1)
    error_norm = torch.mean(torch.abs(error * mask.float()))
    masked_target= torch.mean(torch.abs(target * mask.float()))
    l1_error = error_norm/masked_target
    return l1_error
def plot_error_distributions(m,seed = 420, load_path=None):
    """
       Plot error distributions for the model on validation data.

       Args:
           m: trained model
           seed: random seed for reproducibility
           load_path: path to load model weights
       """
    setup_seed(seed)
    x_errors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x_train = batch['x'].to(device)
            y_train = batch['y'].to(device)
            z_train = batch['z'].to(device)
            mask_train = batch['mask'].to(device)
            t_train = batch['time'].to(device)
            fx_train = batch['fx'].to(device)
            z_pred= m.forward(x = x_train,y=y_train,t=t_train,fx=fx_train)
            loss= masked_l1(z_pred,z_train,mask_train)
            x_errors.extend(loss.cpu().numpy().flatten())
    x_errors = np.array(x_errors)
    x_mean = x_errors.mean()
    print(f'phi loss mean: {x_mean}')
    plt.figure(figsize=(6, 3))

    plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label size
    plt.rcParams['ytick.labelsize'] = 16

    sns.kdeplot(x_errors, fill=True, alpha=0.3, color='blue',
                linewidth=2, log_scale=False, label='Density')

    plt.axvline(x_mean, color='black', linestyle='--', linewidth=3,
                label=f'Mean: {x_mean:.4e}')

    # plt.xlim(2e-3, 1e-1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([])
    plt.gca().set_ylabel('')
    plt.tight_layout()
    plt.savefig('.\\hl1.svg', bbox_inches='tight', transparent=True)
    plt.show()


from torch.cuda.amp import autocast
import psutil
import GPUtil
# from thop import profile
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional


def measure_model_efficiency(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_warmup: int = 10,
        num_iterations: int = 100,
        batch_size: Optional[int] = None,
        use_amp: bool = True,
        profile_flops: bool = True
) -> Dict[str, Any]:
    """
    Measure model efficiency metrics including inference time, memory usage, FLOPs, etc.

    Args:
        model: PyTorch model to measure
        val_loader: data loader for validation
        device: computation device
        num_warmup: number of warmup iterations
        num_iterations: number of measurement iterations
        batch_size: specified batch size, if None use loader's batch size
        use_amp: whether to use automatic mixed precision
        profile_flops: whether to calculate FLOPs

    Returns:
        Dictionary containing various metrics
    """
    model.eval()
    model.to(device)

    # Get a sample batch
    data_iter = iter(val_loader)
    batch = next(data_iter)

    # Prepare input data
    inputs = {
        'x': batch['x'].to(device),
        'y': batch['y'].to(device),
        't': batch['time'].to(device),
        'fx': batch['fx'].to(device)
    }

    # Use specified batch size if provided
    if batch_size is not None and batch_size != inputs['x'].size(0):
        inputs = {k: v[:batch_size] for k, v in inputs.items()}

    results = {}

    # 1. Measure inference time
    print("Starting inference time measurement...")
    inference_times = []

    # Warmup iterations
    with torch.no_grad():
        if use_amp:
            with autocast():
                for _ in range(num_warmup):
                    _ = model.forward(**inputs)
        else:
            for _ in range(num_warmup):
                _ = model.forward(**inputs)

    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Formal measurement
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()

            if use_amp:
                with autocast():
                    _ = model.forward(**inputs)
            else:
                _ = model.forward(**inputs)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Time statistics
    times_ms = np.array(inference_times)
    results['inference_time'] = {
        'mean_ms': float(times_ms.mean()),
        'std_ms': float(times_ms.std()),
        'min_ms': float(times_ms.min()),
        'max_ms': float(times_ms.max()),
        'median_ms': float(np.median(times_ms)),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'p99_ms': float(np.percentile(times_ms, 99)),
        'fps': 1000 / float(times_ms.mean()) if times_ms.mean() > 0 else 0
    }

    # 2. Measure GPU memory
    if device.type == 'cuda':
        print("Measuring GPU memory usage...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Initial memory
        initial_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB

        # One inference to record peak memory
        with torch.no_grad():
            if use_amp:
                with autocast():
                    _ = model.forward(**inputs)
            else:
                _ = model.forward(**inputs)

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB
        allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB

        results['gpu_memory_mb'] = {
            'initial': initial_memory,
            'peak': peak_memory,
            'allocated': allocated_memory,
            'inference_peak': peak_memory - initial_memory
        }

        # Get GPU information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                results['gpu_info'] = {
                    'name': gpu.name,
                    'total_memory_mb': gpu.memoryTotal,
                    'free_memory_mb': gpu.memoryFree,
                    'utilization': gpu.load
                }
        except:
            pass

    # 3. Measure CPU memory
    process = psutil.Process()
    initial_cpu_memory = process.memory_info().rss / 1024 ** 2  # MB

    with torch.no_grad():
        if use_amp:
            with autocast():
                _ = model.forward(**inputs)
        else:
            _ = model.forward(**inputs)

    peak_cpu_memory = process.memory_info().rss / 1024 ** 2  # MB

    results['cpu_memory_mb'] = {
        'initial': initial_cpu_memory,
        'peak': peak_cpu_memory,
        'inference_increase': peak_cpu_memory - initial_cpu_memory
    }

    # 4. Calculate FLOPs and parameters
    if profile_flops:
        print("Calculating FLOPs and parameters...")
        try:
            # Use thop to calculate FLOPs
            from thop import clever_format, profile

            dummy_inputs = (
                inputs['x'][:1],  # Use batch size=1 for calculation
                inputs['y'][:1],
                inputs['t'][:1],
                inputs['fx'][:1]
            )

            flops, params = profile(model, inputs=dummy_inputs, verbose=False)
            flops, params = clever_format([flops, params], "%.3f")

            results['complexity'] = {
                'flops': flops,
                'params': params,
                'flops_raw': int(flops.replace('G', 'e9').replace('M', 'e6').replace('K', 'e3')),
                'params_raw': int(params.replace('G', 'e9').replace('M', 'e6').replace('K', 'e3'))
            }
        except Exception as e:
            print(f"Error calculating FLOPs: {e}")
            results['complexity'] = None

    # 5. Batch size scalability test
    print("Testing performance with different batch sizes...")
    if batch_size is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    else:
        batch_sizes = [batch_size]

    batch_results = {}
    for bs in batch_sizes:
        if bs <= inputs['x'].size(0):
            # Create input for current batch size
            bs_inputs = {k: v[:bs] for k, v in inputs.items()}

            # Measure inference time
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            with torch.no_grad():
                if use_amp:
                    with autocast():
                        _ = model.forward(**bs_inputs)
                else:
                    _ = model.forward(**bs_inputs)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = (time.time() - start_time) * 1000  # milliseconds

            batch_results[f'batch_{bs}'] = {
                'time_ms': inference_time,
                'fps': 1000 / inference_time if inference_time > 0 else 0,
                'throughput': bs * 1000 / inference_time if inference_time > 0 else 0
            }

    results['batch_scaling'] = batch_results

    return results


def print_efficiency_report(results: Dict[str, Any]):
    """Print efficiency measurement report"""
    print("\n" + "=" * 60)
    print("Model Efficiency Analysis Report")
    print("=" * 60)

    # Inference time
    if 'inference_time' in results:
        time_info = results['inference_time']
        print(f"\n📊 Inference Time (batch size={1}):")
        print(f"  Average time: {time_info['mean_ms']:.2f} ms")
        print(f"  FPS: {time_info['fps']:.1f}")
        print(f"  Standard deviation: ±{time_info['std_ms']:.2f} ms")
        print(f"  Median: {time_info['median_ms']:.2f} ms")
        print(f"  P95: {time_info['p95_ms']:.2f} ms")
        print(f"  P99: {time_info['p99_ms']:.2f} ms")

    # GPU memory
    if 'gpu_memory_mb' in results:
        mem_info = results['gpu_memory_mb']
        print(f"\n🎮 GPU Memory Usage:")
        print(f"  Initial memory: {mem_info['initial']:.1f} MB")
        print(f"  Peak memory: {mem_info['peak']:.1f} MB")
        print(f"  Inference increase: {mem_info['inference_peak']:.1f} MB")

    # CPU memory
    if 'cpu_memory_mb' in results:
        cpu_info = results['cpu_memory_mb']
        print(f"\n💻 CPU Memory Usage:")
        print(f"  Initial memory: {cpu_info['initial']:.1f} MB")
        print(f"  Peak memory: {cpu_info['peak']:.1f} MB")

    # Complexity
    if 'complexity' in results and results['complexity']:
        comp_info = results['complexity']
        print(f"\n⚙️ Model Complexity:")
        print(f"  FLOPs: {comp_info['flops']}")
        print(f"  Parameters: {comp_info['params']}")

    # Batch scalability
    if 'batch_scaling' in results:
        print(f"\n📈 Batch Size Scalability:")
        for batch_size, info in results['batch_scaling'].items():
            bs = int(batch_size.split('_')[1])
            print(f"  Batch Size {bs:2d}: {info['time_ms']:6.2f} ms | "
                  f"FPS: {info['fps']:6.1f} | "
                  f"Throughput: {info['throughput']:6.1f} samples/s")

    # GPU information
    if 'gpu_info' in results:
        gpu_info = results['gpu_info']
        print(f"\n🖥️ GPU Information:")
        print(f"  Model: {gpu_info['name']}")
        print(f"  Total memory: {gpu_info['total_memory_mb']:.0f} MB")
        print(f"  Utilization: {gpu_info['utilization']:.1%}")

    print("=" * 60)


def plot_efficiency_metrics(results: Dict[str, Any], save_path: str = None):
    """Visualize efficiency metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Inference time distribution
    if 'inference_time' in results:
        ax = axes[0, 0]
        # Create simulated time distribution data
        mean_time = results['inference_time']['mean_ms']
        std_time = results['inference_time']['std_ms']
        times = np.random.normal(mean_time, std_time, 1000)

        sns.histplot(times, kde=True, ax=ax, color='skyblue')
        ax.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}ms')
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Inference Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Batch scalability
    if 'batch_scaling' in results:
        ax = axes[0, 1]
        batch_sizes = []
        throughputs = []
        fps_values = []

        for batch_str, info in results['batch_scaling'].items():
            bs = int(batch_str.split('_')[1])
            batch_sizes.append(bs)
            throughputs.append(info['throughput'])
            fps_values.append(info['fps'])

        ax.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8, label='Throughput')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Samples/s')
        ax.set_title('Throughput vs Batch Size')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add second y-axis for FPS
        ax2 = ax.twinx()
        ax2.plot(batch_sizes, fps_values, 's-', color='orange', linewidth=2, markersize=6, label='FPS')
        ax2.set_ylabel('FPS')
        ax2.legend(loc='upper left')

    # 3. Memory usage
    if 'gpu_memory_mb' in results:
        ax = axes[1, 0]
        mem_info = results['gpu_memory_mb']
        categories = ['Initial', 'Peak', 'Inference']
        values = [mem_info['initial'], mem_info['peak'], mem_info['inference_peak']]

        bars = ax.bar(categories, values, color=['lightblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel('Memory (MB)')
        ax.set_title('GPU Memory Usage')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom')

    # 4. Time statistics summary
    if 'inference_time' in results:
        ax = axes[1, 1]
        time_info = results['inference_time']
        metrics = ['Mean', 'Median', 'P95', 'P99']
        values = [time_info['mean_ms'], time_info['median_ms'],
                  time_info['p95_ms'], time_info['p99_ms']]

        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylabel('Time (ms)')
        ax.set_title('Inference Time Statistics')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")

    plt.show()

if __name__ == '__main__':
    npz_file_path = r"\iso_npz.npz"
    seed = 58
    train_loader, val_loader = create_data_loaders(npz_file_path, batch_size=10,seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_path = "\\H_model.pth"

    # plot_error_distributions( embed_dim=256, load_path=load_path,seed=seed)
    # setup_seed(seed)
    # m = Model(embed_dim=256, device=device,phi_model_path="\phi_first_model.pth")
    # m.load_state_dict(torch.load(load_path))
    # m = m.to('cuda')

    setup_seed(seed)
    m = Model(embed_dim=256, device=device, phi_model_path="\phi_first_model.pth")
    m.load_state_dict(torch.load(load_path))
    m = m.to('cuda')

    # Measure model efficiency
    print("Starting model efficiency measurement...")
    efficiency_results = measure_model_efficiency(
        model=m,
        val_loader=val_loader,
        device=device,
        num_warmup=5,
        num_iterations=50,
        use_amp=True,
        profile_flops=True
    )

    # Print report
    print_efficiency_report(efficiency_results)

    # Plot charts
    plot_efficiency_metrics(efficiency_results, save_path='model_efficiency_analysis.png')
    # plot_error_distributions(m)
