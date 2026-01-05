import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Sampler
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


class AdvancedBucketSampler(Sampler):
    """
      Sampler that groups sequences of similar lengths into buckets to minimize padding.

      This reduces computational waste by ensuring sequences in the same batch have
      similar lengths, which is especially useful for variable-length sequence data.
      """
    def __init__(self, dataset, batch_size, bucket_method='quantile',
                 num_buckets=10, shuffle=True, drop_last=False, seed=42):

        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_method = bucket_method
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.lengths = self._get_sequence_lengths()

        self.buckets, self.bucket_boundaries = self._create_buckets()


    def _get_sequence_lengths(self):
        """Extract sequence lengths for all samples in the dataset."""
        lengths = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            lengths.append(len(sample['x']))
        return lengths

    def _create_buckets(self):
        """
               Create buckets based on sequence lengths using the specified method.

               Returns:
                   buckets: Dictionary mapping bucket indices to sample indices
                   boundaries: Array of bucket boundaries
               """
        lengths = np.array(self.lengths)
        # Different methods for determining bucket boundaries
        if self.bucket_method == 'quantile':
            # Use quantiles to create buckets with equal sample counts
            percentiles = np.linspace(0, 100, self.num_buckets + 1)
            boundaries = np.percentile(lengths, percentiles)

        elif self.bucket_method == 'uniform':
            # Uniformly spaced buckets between min and max lengths
            min_len, max_len = min(lengths), max(lengths)
            boundaries = np.linspace(min_len, max_len, self.num_buckets + 1)

        elif self.bucket_method == 'geometric':
            # Geometrically spaced buckets (logarithmic scale)

            min_len, max_len = min(lengths), max(lengths)
            boundaries = np.geomspace(min_len, max_len, self.num_buckets + 1)

        else:
            raise ValueError(f"Unknown bucket method: {self.bucket_method}")

        boundaries = np.unique(np.round(boundaries).astype(int))

        buckets = defaultdict(list)
        for idx, length in enumerate(lengths):
            bucket_idx = np.searchsorted(boundaries, length, side='right') - 1
            bucket_idx = max(0, min(bucket_idx, len(boundaries) - 2))
            buckets[bucket_idx].append(idx)

        return dict(buckets), boundaries

    def _print_bucket_info(self):
        """Print statistical information about the created buckets."""
        print("=== Bucket Statistics ===")
        print(f"Total samples: {len(self.dataset)}")
        print(f"Number of buckets: {len(self.buckets)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Bucket method: {self.bucket_method}")
        print("\nDetailed bucket information:")

        for bucket_idx in sorted(self.buckets.keys()):
            samples = self.buckets[bucket_idx]
            if not samples:
                continue

            bucket_lengths = [self.lengths[i] for i in samples]
            min_len = min(bucket_lengths)
            max_len = max(bucket_lengths)
            avg_len = np.mean(bucket_lengths)

            print(f"Bucket {bucket_idx}: {len(samples)} samples, "
                  f"length range [{min_len}, {max_len}], "
                  f"average length {avg_len:.1f}, "
                  f"can form {len(samples) // self.batch_size} full batches")


    def _create_batches(self):
        all_batches = []

        for bucket_idx, indices in self.buckets.items():
            if len(indices) == 0:
                continue

            if self.shuffle:
                rng = np.random.RandomState(self.seed + bucket_idx)
                rng.shuffle(indices)

            batches = []
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if not self.drop_last or len(batch_indices) == self.batch_size:
                    batches.append(batch_indices)

            all_batches.extend(batches)
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(all_batches)

        return all_batches

    def __iter__(self):
        batches = self._create_batches()
        return iter(batches)

    def __len__(self):
        total_batches = 0
        for indices in self.buckets.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total_batches += n_batches
        return total_batches

class IsoSurfaceDataset(Dataset):
    def __init__(self, npz_file, transform=None):

        data = np.load(npz_file, allow_pickle=True)
        self.structured_data = data['structured_data']
        self.transform = transform

        print(f"Loaded dataset with {len(self.structured_data)} samples")
    def __len__(self):
        return len(self.structured_data)

    def __getitem__(self, idx):
        sample = self.structured_data[idx]
        kw = torch.tensor(sample['kw'], dtype=torch.float32)
        ko = torch.tensor(sample['ko'], dtype=torch.float32)
        time_point = torch.tensor(sample['time'], dtype=torch.float32)
        data_tensor = torch.tensor(sample['data'], dtype=torch.float32)
        fx = torch.FloatTensor([kw, ko])

        x = data_tensor[:, 1]
        y = data_tensor[:, 0]
        z = data_tensor[:, 2]

        return {
            'x': x,
            'y': y,
            'z': z,
            'fx': fx,
            'time': time_point
        }


def optimized_collate_fn(batch):

    x_list = [item['x'] for item in batch]
    y_list = [item['y'] for item in batch]
    z_list = [item['z'] for item in batch]

    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)
    z_padded = pad_sequence(z_list, batch_first=True, padding_value=0.0)

    lengths = torch.tensor([len(x) for x in x_list])
    max_len = x_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    param_list = [item['fx'] for item in batch]
    fx = torch.stack(param_list)
    time = torch.FloatTensor([item['time'] for item in batch])

    return {
        'x': x_padded.unsqueeze(-1),  # (batch, seq_len, 1)
        'y': y_padded.unsqueeze(-1),
        'z': z_padded,
        'fx': fx,
        'time': time.unsqueeze(-1),
        'mask': mask,
    }


def create_data_loaders(npz_file, batch_size=32, train_ratio=0.8, shuffle=True,seed = 42,num_buckets=10, bucket_method='geometric',num_workers=4,):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dataset = IsoSurfaceDataset(npz_file)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size


    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_sampler = AdvancedBucketSampler(
        train_dataset,
        batch_size=batch_size,
        num_buckets=num_buckets,
        bucket_method=bucket_method,
        shuffle=shuffle,
        seed=seed
    )

    val_sampler = AdvancedBucketSampler(
        val_dataset,
        batch_size=batch_size,
        num_buckets=num_buckets,
        bucket_method=bucket_method,
        shuffle=False,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=optimized_collate_fn,
        pin_memory= True,
        persistent_workers= True,
        prefetch_factor= 1,
        num_workers= 4,
        pin_memory_device='cuda'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=optimized_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
        num_workers=4,
        pin_memory_device='cuda'

    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


def analyze_padding_efficiency(data_loader, num_batches=10):
    """
    Analyze the padding efficiency of the data loader.

    This helps evaluate how effective the bucket sampling is at reducing
    wasted computation due to padding.
    """
    start_time = time.time()

    total_elements = 0
    padding_elements = 0

    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break

        batch_size, seq_len = batch['x'].shape[:2]
        total_elements += batch_size * seq_len

        # Calculate padding elements (assuming 'lengths' is in the batch)
        # Note: The original code references batch['lengths'] but it's not in the collate_fn
        # This needs to be adjusted based on actual batch structure
        if 'lengths' in batch:
            for length in batch['lengths']:
                padding_elements += (seq_len - length)
        else:
            # Estimate from mask if lengths not available
            mask = batch.get('mask', None)
            if mask is not None:
                padding_elements += ((~mask).sum().item())
            else:
                print("Warning: Cannot calculate padding without lengths or mask")
                break

        padding_ratio = padding_elements / total_elements if total_elements > 0 else 0

        print(f"Batch {i + 1}: Padding ratio = {padding_ratio:.2%}, Sequence length = {seq_len}")

    if total_elements > 0:
        avg_padding_ratio = padding_elements / total_elements
        print(f"\nAverage padding ratio: {avg_padding_ratio:.2%}")
    else:
        print("\nNo batches processed or unable to calculate padding ratio")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Code execution time: {elapsed_time:.2f} seconds")
if __name__ == "__main__":
    npz_file_path = r"D:\desktop\iso_npz.npz"

    train_loader, val_loader = create_data_loaders(
        npz_file_path,
        batch_size=50,
        train_ratio=0.8
    )

    print("\nTesting data loaders:")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print(f"  - Number of point clouds: {batch['x'].shape}")
        print(f"  - Parameters shape: {batch['time'].shape}")
        print(f"  - First point cloud shape: {batch['fx'].shape}")
        break

