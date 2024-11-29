import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from torch.optim import AdamW
from dataset import ADNIDatasetSlicesInOrder
from torch.utils.data import DataLoader
from utils import sample_16_indices, indices_to_mask, save_tloss_csv, load_or_initialize_training
import numpy as np
import time