import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_num = 5
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from torch.optim import AdamW
from common.dataset import ADNIDatasetSlicesInOrder
from torch.utils.data import DataLoader
from common.utils import load_trained_model, load_trained_embedder, plot_batch_slices, normalise_intensity_per_slice, plot_brain_3_views
import numpy as np
import matplotlib.pyplot as plt
from common.sample import sample_iteratively_no_conditioning_and_save
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8
num_train_timesteps = 1000
data_size = 64
symmetric_normalisation = False
index_encoding_type='index_normalised'

slices_at_once = 4

in_dir = "/cim/ehoney/ecse626proj/experiments/experiment14"
best_loss_ckpt_path = os.path.join(in_dir, "best_loss_ckpt_below1500.pth.tar")
out_dir = os.path.join(in_dir, '1500epochs')
os.makedirs(out_dir, exist_ok=True)
out_data_dir = os.path.join(out_dir, 'generated_data')
os.makedirs(out_data_dir, exist_ok=True)
out_plots_dir = os.path.join(out_dir, 'plots')
os.makedirs(out_plots_dir, exist_ok=True)

print("---------------------------------------------------")
print("Initialising model...")
model = UNet2DConditionModel(
    sample_size=data_size,
    in_channels=2*slices_at_once, #16
    out_channels=slices_at_once, #8 #initialise to maximum and compute loss only for indexed slices
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D"),
    mid_block_type=None,
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    # not sure about what the lower arguments do
    block_out_channels=(256, 256, 512, 512, 1024, 1024), #start with 64? no deeper than 1024
    layers_per_block=2, #default
    cross_attention_dim=1, # ~concat one-hot encodings of condition and target slices~ NO now instead just one 128 vector with labels 0,1,2
    attention_head_dim=8, #default
    norm_num_groups=32, #default
    use_linear_projection=True #DIFFERENT
)
print(f"Model initialised. Number of model parameters: {model.num_parameters()}.")
model.to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

epoch = load_trained_model(model, best_loss_ckpt_path)

iterations = 16 # should give ~500 brains

model.eval()
for i in range(iterations):
    print(f"Beginning iteration {i}...")
    start_time = time.time()
    output_batch = sample_iteratively_no_conditioning_and_save(
        slices_at_once=slices_at_once,
        model=model,
        noise_scheduler=noise_scheduler,
        embedder=None,
        batch_size=batch_size,
        device=device,
        symmetric_normalisation=symmetric_normalisation,
        index_encoding_type=index_encoding_type,
        data_size=data_size
    )
    end_time = time.time()
    duration = end_time - start_time
    print(f"Iteration sampling complete. This took {duration} seconds.")
    
    print(f"Saving and plotting...")
    for b in range(batch_size):
        data_name = f'{{GPU={gpu_num}}}_{{iteration={i}}}_{{batch={b}}}'
        np.save(os.path.join(out_data_dir, data_name), output_batch[b])
        # plot_brain_3_views(output_batch[i], os.path.join(out_plots_dir, data_name))
    print("Complete.")