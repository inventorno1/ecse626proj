import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from torch.optim import AdamW
from dataset import ADNIDatasetSlicesInOrder
from torch.utils.data import DataLoader
from utils import sample_16_indices, indices_to_mask, save_tloss_csv, load_or_initialize_training
import numpy as np
import time

# from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
epochs = 1500 # should be 1736.1 to match paper
backup_interval = 10
lr = 1e-4
batch_size = 4
num_train_timesteps = 256
loss_fn = torch.nn.MSELoss()

slices_at_once = 8

out_dir = "/cim/ehoney/ecse626proj/experiment5"
os.makedirs(out_dir, exist_ok=True)
train_loss_path = os.path.join(out_dir, "train_loss.csv")
latest_ckpt_path = os.path.join(out_dir, "latest_ckpt.pth.tar")
backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
os.makedirs(backup_ckpts_dir, exist_ok=True)

print("---------------------------------------------------")
print("Initialising model...")
model = UNet2DConditionModel(
    sample_size=128,
    in_channels=2*slices_at_once, #16
    out_channels=slices_at_once, #8 #initialise to maximum and compute loss only for indexed slices
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    mid_block_type=None,
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    # not sure about what the lower arguments do
    block_out_channels=(64, 128, 256, 512, 1024), #start with 64? no deeper than 1024
    layers_per_block=2, #default
    # cross_attention_dim=256, # ~concat one-hot encodings of condition and target slices~ NO now instead just one 128 vector with labels 0,1,2
    attention_head_dim=8, #default
    norm_num_groups=32, #default
    use_linear_projection=True #DIFFERENT
)
print(f"Model initialised. Number of model parameters: {model.num_parameters()}.")
model.to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
optimizer = AdamW(model.parameters(), lr=lr)
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

print("Searching for checkpoint...")
epoch_start = load_or_initialize_training(model, optimizer, latest_ckpt_path)
print("---------------------------------------------------")

data_dir = "/cim/ehoney/ecse626proj/preprocessed_data"
train_dataset = ADNIDatasetSlicesInOrder(data_dir, n=slices_at_once)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8) # CHANGE to NUM_WORKERS=4

for epoch in range(epoch_start, epochs):
    print(f"Beginning Epoch {epoch + 1}...")

    losses_over_epoch = []

    model.train()
    # s_time = time.time()
    for step, batch in enumerate(train_dataloader):
        # t_time = time.time()
        # print(time.time()-s_time)
        target_slices, condition_slices = batch

        # print("Slices shapes:", target_slices.shape, condition_slices.shape)

        target_slices = target_slices.to(device)
        condition_slices = condition_slices.to(device)

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        # print(f"Timesteps shape: {timesteps.shape}")
        noise = torch.randn_like(target_slices, device=device, dtype=torch.float32)
        # print(f"Noise dtype: {noise.dtype}; timesteps dtype: {timesteps.dtype}")
        noised_target_slices = noise_scheduler.add_noise(target_slices, noise, timesteps)
        # print(f"noised_target_slices dtype: {noised_target_slices.dtype}")

        # Then I need to mask out noisy images to only include condition slices i.e. non-condition slices should be zero - NOT RIGHT, see below
        # need to noise the target slices and feed in the condition slices as condition, along with the index encodings for the attention layer
        
        timesteps = timesteps.float()

        input_tensor = torch.concat((noised_target_slices, condition_slices), dim=1)

        input_tensor = input_tensor.to(torch.float32)
        timesteps = timesteps.to(torch.float32)

        # print("Model arg shapes:", input_tensor.shape, timesteps.shape) #, indices_encoding.shape)
        # print("Model arg dtypes:", input_tensor.dtype, timesteps.dtype, indices_encoding.dtype)
        predicted_noise = model(input_tensor, timesteps, None).sample
        # print("Predicted noise", predicted_noise.shape)

        # print("Noise devices", predicted_noise.device, noise.device)
        loss = loss_fn(predicted_noise, noise)
        # print("Loss device", loss.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")
        losses_over_epoch.append(loss.detach().cpu())
        # break # for debugging, just use one batch
        # s_time = time.time()
        # print(time.time()-t_time)

    average_epoch_loss = np.mean(losses_over_epoch)
    save_tloss_csv(train_loss_path, epoch+1, average_epoch_loss)
    print(f"Epoch {epoch + 1} completed. Average loss = {average_epoch_loss:.4f}")

    print('Saving model checkpoint...')
    checkpoint = {
        'epoch': epoch,
        'model_sd': model.state_dict(),
        'optim_sd': optimizer.state_dict(),
        'loss': average_epoch_loss
    }
    torch.save(checkpoint, latest_ckpt_path)
    if (epoch + 1) % backup_interval == 0:
        torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch + 1}.pth.tar'))
    print('Checkpoint saved successfully.')