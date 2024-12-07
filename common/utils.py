import numpy as np
import lz4.frame
from io import BytesIO
import random
import torch
import csv
import os
import matplotlib.pyplot as plt

def load_from_lz4(path):
    with lz4.frame.open(path, 'rb') as f:
        decompressed_data = f.read()
    decompressed_file = BytesIO(decompressed_data)
    data = np.load(decompressed_file)
    return data

def normalise_intensity(img, lower=1, upper=99):
    lower_bound, upper_bound = np.percentile(img, (lower,upper))
    clipped_img = np.clip(img, lower_bound, upper_bound)
    normalised_img = (clipped_img - lower_bound) / (upper_bound - lower_bound)
    return normalised_img

def normalise_intensity_per_slice(batch_of_slices):
    # Do I still need to clip here??

    # batch_of_slices has shape (B, C, 128, 128)
    # below have shape (B,C, 1, 1)
    min_vals = torch.amin(batch_of_slices, dim=(2,3), keepdim=True)
    max_vals = torch.amax(batch_of_slices, dim=(2,3), keepdim=True)
    normalised_batch_of_slices = (batch_of_slices - min_vals) / (max_vals - min_vals + 1e-8)
    return normalised_batch_of_slices

def sample_indices(max_slices=20, num_slices=128):

    total_indices = random.randint(1, max_slices)

    indices = random.sample(range(1,num_slices), total_indices)

    target_size = random.randint(1, len(indices))
    target = random.sample(indices, target_size)
    condition = [index for index in indices if index not in target]

    # Encode indices
    encoding = np.zeros(num_slices)
    encoding[condition] = 1
    encoding[target] = 2

    return condition, target, encoding

def indices_to_mask(indices, size=128):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[indices] = True
    return mask

def sample_16_indices(num_slices=128, include_condition_prob = 0.8):

    include_condition = False
    if random.random() < include_condition_prob:
        include_condition = True

    if include_condition:
        indices = random.sample(range(1,num_slices), 16)
        target = random.sample(indices, 8)
        condition = [index for index in indices if index not in target]
    else:
        target = random.sample(range(1,num_slices), 8)
        condition = []

    target_encoding = np.zeros(num_slices)
    target_encoding[target] = 1
    condition_encoding = np.zeros(num_slices)
    condition_encoding[condition] = 1
    encoding = np.concatenate((target_encoding, condition_encoding))

    return target, condition, encoding


def save_tloss_csv(pathname, epoch, tloss):
    with open(pathname, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            writer.writerow(['Epoch', 'Training Loss'])
        writer.writerow([epoch, tloss])

def load_or_initialize_training(model, optimizer, latest_ckpt_path, get_best_loss=False):

    if not os.path.exists(latest_ckpt_path):
        epoch_start = 0
        print('No training checkpoint found. Will start training from scratch.')
        if get_best_loss:
            return epoch_start, float('inf')
    else:
        print('Training checkpoint found. Loading checkpoint...')
        checkpoint = torch.load(latest_ckpt_path)
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_sd'])
        optimizer.load_state_dict(checkpoint['optim_sd'])
        print(f'Checkpoint loaded from epoch {epoch_start - 1 + 1}. Will continue training from epoch {epoch_start + 1}.')

        if get_best_loss:
            best_loss = checkpoint['best_loss']
            return epoch_start, best_loss

    return epoch_start

def load_trained_model(model, ckpt_path):

    checkpoint = torch.load(ckpt_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_sd'])
    print(f'Checkpoint loaded from epoch {epoch + 1}.')

    return epoch

def load_trained_embedder(embedder, ckpt_path):

    checkpoint = torch.load(ckpt_path)
    epoch = checkpoint['epoch']
    embedder.load_state_dict(checkpoint['embedder_sd'])
    print(f'Checkpoint loaded from epoch {epoch + 1}.')

    return None


def plot_batch_slices(input_tensor, batch_size, save_path=None, slices=8, vmin=0, vmax=1):

    # Create a grid of subplots with rows for batches and columns for channels
    fig, axes = plt.subplots(batch_size, slices, figsize=(slices * 2, batch_size * 2))

    # # Determine the global min and max values for the color scale
    # vmin = input_tensor.min().item()
    # vmax = input_tensor.max().item()

    # Loop through each batch and channel to plot
    for b in range(batch_size):
        for c in range(slices):
            ax = axes[b, c] if batch_size > 1 else axes[c]  # Handle 1D subplot case
            im = ax.imshow(input_tensor[b, c].cpu(), cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')  # Optional: Remove axis labels for cleaner visualization
            if b == 0:
                ax.set_title(f'Slices {c+1}')  # Set channel title on the first row

    # Add a universal colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Intensity')

    # Add a global title or adjust layout
    # plt.suptitle('Batch and Channel Visualization with Universal Colorbar', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    return None


def plot_brain_3_views(brain_data, save_path=None):

    x, y, z = brain_data.shape
    assert (x == y == z)

    fig, axes = plt.subplots(1, 3, figsize=(6, 2))
    for i, axis in enumerate([0,1,2]):
        axes[i].imshow(brain_data.take(x//2, axis=i), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()