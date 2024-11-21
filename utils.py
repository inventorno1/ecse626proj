import numpy as np
import lz4.frame
from io import BytesIO
import random
import torch

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
    