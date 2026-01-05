import time
from datetime import timedelta
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.G_dataloaders import get_dataloaders
from model.G_model import Model
from utils.loss import setup_seed, LpLoss
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
import pickle

def masked_l2_loss(pred, target, mask):
    num_examples = pred.size()[0]
    error = pred - target
    error = error * mask
    target = (target* mask).reshape(num_examples, -1)
    error_norm = torch.norm(error.reshape(num_examples, -1), p=2)
    masked_target= torch.norm(target, p=2)
    l2_error = torch.sum(error_norm/masked_target)
    return l2_error
def masked_l1_loss(pred, target, mask):
    num_examples = pred.size()[0]
    error = pred - target
    error = error * mask
    target = (target * mask).reshape(num_examples, -1)
    error_norm = torch.norm(error.reshape(num_examples, -1), p=1)
    masked_target = torch.norm(target, p=1)
    l2_error = torch.sum(error_norm / masked_target)
    return l2_error

_cached_train_loader = None
_cached_val_loader = None
_cached_npz_path = None


class DataLoaderCache:
    def __init__(self, cache_dir='./dataloader_cache/'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, npz_file_path, batch_size, seed):
        """Generate unique cache key based on file path, batch size, seed and file modification time"""
        file_mtime = os.path.getmtime(npz_file_path) if os.path.exists(npz_file_path) else 0
        return f"{npz_file_path}_{batch_size}_{seed}_{file_mtime}.pkl"

    def get_dataloaders(self, npz_file_path, batch_size=32, seed=42):
        cache_file = os.path.join(self.cache_dir, self._get_cache_key(npz_file_path, batch_size, seed))

        # Check if cache exists and is valid
        if os.path.exists(cache_file):
            try:
                print("Loading cached data loaders...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError, Exception) as e:
                print(f"Cache loading failed: {e}, creating new data loaders...")

        # Create new data loaders and cache them
        print("Creating new data loaders and caching...")
        train_loader, val_loader = get_dataloaders(npz_file_path, batch_size=batch_size, seed=seed)

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((train_loader, val_loader), f)
            print(f"Data loaders cached to: {cache_file}")
        except Exception as e:
            print(f"Cache saving failed: {e}, but training can continue")

        return train_loader, val_loader
def train(epoches, lr, embed_dim=64, seed=420, log=None, save_dir=None, load_path=None):
    setup_seed(seed)
    m = Model(embed_dim=embed_dim, device=device)
    start_time = time.time()
    epoch_times = []

    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}. Check model architecture or compatibility. Training from scratch.")
    m = m.to('cuda')
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, amsgrad=False, weight_decay=0.1)
    steps_per_epoch = len(train_loader)
    total_steps = epoches * steps_per_epoch

    # OneCycleLR scheduler for learning rate scheduling
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,  # Maximum learning rate
        total_steps=total_steps,  # Total training steps
        pct_start=0.1,  # Percentage of steps for warmup phase
        div_factor=25.0,  # Initial learning rate = max_lr / div_factor
        final_div_factor=1e5  # Final learning rate = max_lr / final_div_factor
    )

    if log is not None:
        writer = SummaryWriter(log_dir=log)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    global_step = 0

    for epoch in tqdm(range(epoches), desc="Training", unit="epoch"):
        epoch_start_time = time.time()
        Train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            x_train = batch['x'].to(device)
            y_train = batch['y'].to(device)
            z_train = batch['z'].to(device)
            v_train = batch['v'].to(device)
            mask_train = batch['mask'].to(device)
            t_train = batch['time'].to(device)
            fx_train = batch['fx'].to(device)

            v_pred = m.forward(x=x_train, y=y_train, z=z_train, t=t_train, fx=fx_train)
            loss = masked_l1_loss(v_pred, v_train, mask_train)
            Train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if log is not None and global_step% 100 == 0:
                writer.add_scalar('Train/step_l2', loss.item(), global_step)
            global_step += 1


        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch} Loss {Train_loss / len(train_loader)} Time: {epoch_time:.2f}s')

        Val_loss_s = 0
        Val_loss_sl1 = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x_val = batch['x'].to(device)
                y_val = batch['y'].to(device)
                z_val = batch['z'].to(device)
                v_val = batch['v'].to(device)
                t_val = batch['time'].to(device)
                mask_val = batch['mask'].to(device)
                fx_val = batch['fx'].to(device)
                v_pred_val = m.forward(x=x_val, y=y_val, z=z_val, t=t_val, fx=fx_val)
                loss_val = masked_l2_loss(v_pred_val, v_val, mask_val)
                loss_l1_val = masked_l1_loss(v_pred_val, v_val, mask_val)
                Val_loss_s += loss_val.item()
                Val_loss_sl1 += loss_l1_val.item()

        Train_loss = Train_loss / len(train_loader)
        Val_loss = Val_loss_s / len(val_loader)
        Val_loss_sl1 = Val_loss_sl1 / len(val_loader)
        if log is not None:
            writer.add_scalar('Train/l2', Train_loss, epoch)
            writer.add_scalar('Val/l2', Val_loss, epoch)
            writer.add_scalar('Val/l1', Val_loss_sl1, epoch)

        if save_dir is not None and epoch % 100 == 0:  # Save every 100 epochs
            save_path = os.path.join(save_dir, f"G_model.pth")
            torch.save(m.state_dict(), save_path)
            print(f"Model saved at {save_path}")

    total_time = time.time() - start_time

    if len(epoch_times) > 5:
        avg_epoch_time = sum(epoch_times[5:]) / len(epoch_times[5:])
    else:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Print training time statistics
    print("\n" + "=" * 50)
    print("Training Time Statistics:")
    print(f"Total training time: {timedelta(seconds=int(total_time))}")
    print(f"Total epochs: {epoches}")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    print(f"Fastest epoch time: {min(epoch_times):.2f} seconds")
    print(f"Slowest epoch time: {max(epoch_times):.2f} seconds")
    print(f"Estimated data loading/processing time: {timedelta(seconds=int(total_time - sum(epoch_times)))}")
    print("=" * 50)
    if log is not None:
        # Log time statistics to TensorBoard
        writer.add_scalar('Time/total_training_time', total_time, 0)
        writer.add_scalar('Time/avg_epoch_time', avg_epoch_time, 0)
        writer.add_scalar('Time/min_epoch_time', min(epoch_times), 0)
        writer.add_scalar('Time/max_epoch_time', max(epoch_times), 0)
        writer.close()



if __name__ == '__main__':
    npz_file_path = r"all_data_15000.npz"
    seed = 42
    cache = DataLoaderCache()
    train_loader, val_loader = cache.get_dataloaders(npz_file_path, batch_size=32, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = './Checkpoint/'
    os.makedirs(save_dir, exist_ok=True)
    base_log_dir = ('/root/tf-logs/12_4_A')
    load_path = "D:\desktop\BCO\checkpoint\water\phi_first_model.pth"
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    time_based_unique_id = f"{current_time}"
    log_dir = os.path.join(base_log_dir, "Gmodel---" + time_based_unique_id)
    save_dir = os.path.join(save_dir, "Gmodel---" + time_based_unique_id)
    train(500, lr = 1e-3, embed_dim=128, log=log_dir, save_dir=save_dir, load_path=None,seed=seed)

