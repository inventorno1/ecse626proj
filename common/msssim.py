# This code is derived from the MRIAnatEval repository: 
# https://github.com/jiaqiw01/MRIAnatEval
# All rights belong to the original authors.

from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from common.pytorch_ssim import msssim_3d
import os

# ADAPTED FROM https://github.com/jiaqiw01/MRIAnatEval/blob/main/evaluation.py
def msssim_real(real_samples: list, device):
    '''
        Calculating msssim for real volumes
    '''
    real_ssim = 0
    for k in range(1000):
        print(f'Iteration {k}...')
        idx1, idx2 = np.random.choice(np.arange(len(real_samples)), size=2, replace=False)
        # Get 2 real volumes
        vol1 = torch.from_numpy(real_samples[idx1]).unsqueeze(0).unsqueeze(0).to(device)
        vol2 = torch.from_numpy(real_samples[idx2]).unsqueeze(0).unsqueeze(0).to(device)
        msssim = msssim_3d(vol1,vol2, normalize=False) #no need to normalise as model does this
        real_ssim  += msssim
    avg_real_ssim = real_ssim/(k+1)
    print("Real volume avg ssim: ", avg_real_ssim)
    return avg_real_ssim