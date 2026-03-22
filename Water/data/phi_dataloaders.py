import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


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
        input_data = data_tensor[:, 1]
        target_data = data_tensor[:, 0]

        return {
            'input': input_data,
            'target': target_data,
            'fx': fx,

            'time': time_point
        }


def collate_fn(batch):

    input_list = [item['input'] for item in batch]
    target_list = [item['target'] for item in batch]
    param_list = [item['fx'] for item in batch]
    fx = torch.stack(param_list)

    time = torch.FloatTensor([item['time'] for item in batch])

    max_length = max(data.shape[0] for data in input_list)

    padded_inputs = []
    for input_data in input_list:
        current_length = input_data.shape[0]

        if current_length < max_length:
            repeated_input = input_data.repeat(2)
            padded_input = repeated_input[:max_length]
        else:
            padded_input = input_data[:max_length]

        padded_inputs.append(padded_input.unsqueeze(1))

 
    padded_targets = []
    for target_data in target_list:
        current_length = target_data.shape[0]

        if current_length < max_length:
            repeated_target = target_data.repeat(2)
            padded_target = repeated_target[:max_length]
        else:
            padded_target = target_data[:max_length]

        padded_targets.append(padded_target)

    input_tensor = torch.stack(padded_inputs)
    target_tensor = torch.stack(padded_targets)

    return {
        'input': input_tensor,
        'target': target_tensor,
        'fx': fx,
        'time': time.unsqueeze(-1)
    }
def create_data_loaders(npz_file, batch_size=32, train_ratio=0.8, shuffle=True,seed = 42):
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator if shuffle else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


