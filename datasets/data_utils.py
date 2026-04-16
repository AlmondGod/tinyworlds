import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.datasets import PongDataset, SonicDataset, PolePositionDataset, PicoDoomDataset, ZeldaDataset, StreetFighterDataset, TerrariaDataset, SpaceInvadersDataset

DEFAULT_NUM_WORKERS = 2
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_PIN_MEMORY = False
DEFAULT_PERSISTENT_WORKERS = True


def _default_video_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def _load_video_dataset_pair(dataset_cls, video_rel_path, h5_rel_path, num_frames, transform=None, fps=30, preload_ratio=1, **kwargs):
    current_folder_path = os.getcwd()
    video_path = current_folder_path + video_rel_path
    preprocessed_path = current_folder_path + h5_rel_path
    transform = _default_video_transform() if transform is None else transform

    train = dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=True,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        **kwargs
    )
    val = dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=False,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        **kwargs
    )
    return train, val


def load_pong(num_frames=1, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        PongDataset,
        '/data/pong.mp4',
        '/data/pong_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_sonic(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        SonicDataset,
        '/data/sonic_frames.mp4',
        '/data/sonic_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_pole_position(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        PolePositionDataset,
        '/data/pole_position.mp4',
        '/data/pole_position_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_picodoom(num_frames=4, fps=30, preload_ratio=1):
    return _load_video_dataset_pair(
        PicoDoomDataset,
        '/data/picodoom cleaned.mp4',
        '/data/picodoom_frames.h5',
        num_frames=num_frames,
        fps=30,
        preload_ratio=preload_ratio
    )


def load_zelda(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        ZeldaDataset,
        '/data/Zelda oot2d 1 Cut.mp4',
        '/data/zelda_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_street_fighter(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        StreetFighterDataset,
        '/data/street_fighter.mp4',
        '/data/street_fighter_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_terraria(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        TerrariaDataset,
        '/data/terraria.mp4',
        '/data/terraria_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_space_invaders(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        SpaceInvadersDataset,
        '/data/space_invaders.mp4',
        '/data/space_invaders_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def data_loaders(train_data, val_data, batch_size, distributed=False, rank=0, world_size=1):
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False if train_sampler is not None else True,
        sampler=train_sampler,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=DEFAULT_PERSISTENT_WORKERS,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False if val_sampler is not None else True,
        sampler=val_sampler,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=DEFAULT_PERSISTENT_WORKERS,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        drop_last=True
    )
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, num_frames=1, distributed=False, rank=0, world_size=1, fps=15, preload_ratio=1):
    if dataset == 'PONG':
        training_data, validation_data = load_pong(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'SONIC':
        training_data, validation_data = load_sonic(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'POLE_POSITION':
        training_data, validation_data = load_pole_position(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'PICODOOM':
        training_data, validation_data = load_picodoom(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'ZELDA':
        training_data, validation_data = load_zelda(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'STREET_FIGHTER':
        training_data, validation_data = load_street_fighter(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'TERRARIA':
        training_data, validation_data = load_terraria(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'SPACE_INVADERS':
        training_data, validation_data = load_space_invaders(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    else:
        raise ValueError('Invalid dataset')

    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size,
        distributed=distributed, rank=rank, world_size=world_size
    )
    x_train_var = np.var(training_data.data)

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def visualize_reconstruction(original, reconstruction, save_path=None):
    # original: (B, C, H, W) or (B, T, C, H, W)
    # reconstruction: (B, C, H, W) or (B, T, C, H, W) 

    # move tensors to CPU and convert to float32 for matplotlib compatibility
    original = original.detach().to('cpu', dtype=torch.float32)
    reconstruction = reconstruction.detach().to('cpu', dtype=torch.float32)

    # handle single frames by expanding to sequences
    if original.dim() == 4:  # (B, C, H, W)
        original = original.unsqueeze(1)  # Add sequence dimension
    if reconstruction.dim() == 4:  # (B, C, H, W)
        reconstruction = reconstruction.unsqueeze(1)  # Add sequence dimension

    # take first 4 sequences, each of length 4 (or available length)
    num_sequences = min(4, original.shape[0])
    seq_length = min(4, original.shape[1])

    original = original[:num_sequences, :seq_length]  # (B, T, C, H, W)
    reconstruction = reconstruction[:num_sequences, :seq_length]  # (B, T, C, H, W)

    # create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # for original sequences
    # reshape to (B * T, C, H, W) for make_grid
    orig_flat = original.reshape(-1, *original.shape[2:])  # (B*T, C, H, W)
    grid_orig = make_grid(orig_flat, nrow=seq_length, normalize=True, padding=2).clamp(0, 1)
    ax1.imshow(grid_orig.permute(1, 2, 0).contiguous().numpy())
    ax1.axis('off')
    ax1.set_title(f'Original Sequences (4 sequences × {seq_length} frames)')

    # for reconstructed sequences
    recon_flat = reconstruction.reshape(-1, *reconstruction.shape[2:])  # (B*T, C, H, W)
    grid_recon = make_grid(recon_flat, nrow=seq_length, normalize=True, padding=2).clamp(0, 1)
    ax2.imshow(grid_recon.permute(1, 2, 0).contiguous().numpy())
    ax2.axis('off')
    ax2.set_title(f'Reconstructed Sequences (4 sequences × {seq_length} frames)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
