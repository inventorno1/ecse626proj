import numpy as np
import lz4.frame
from io import BytesIO
import random
import torch
import csv
import os

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

def load_or_initialize_training(model, optimizer, latest_ckpt_path):

    if not os.path.exists(latest_ckpt_path):
        epoch_start = 0
        print('No training checkpoint found. Will start training from scratch.')
    else:
        print('Training checkpoint found. Loading checkpoint...')
        checkpoint = torch.load(latest_ckpt_path)
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_sd'])
        optimizer.load_state_dict(checkpoint['optim_sd'])
        print(f'Checkpoint loaded from epoch {epoch_start - 1}. Will continue training from epoch {epoch_start}.')

    return epoch_start