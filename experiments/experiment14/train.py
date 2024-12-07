import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 6"
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from torch.optim import AdamW
from common.dataset import ADNIDatasetSlicesInOrder
from torch.utils.data import DataLoader
from common.utils import sample_16_indices, indices_to_mask, save_tloss_csv, load_or_initialize_training
import numpy as np
import time
from accelerate import Accelerator

# from torch.optim.lr_scheduler import CosineAnnealingLR

# accelerator = Accelerator()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = accelerator.device
epochs = 1500 # should be 1736.1 to match paper
backup_interval = 50
lr = 2.5e-5
batch_size = 16
num_train_timesteps = 1000
loss_fn = torch.nn.MSELoss()
data_size = 64

slices_at_once = 4

out_dir = "/cim/ehoney/ecse626proj/experiments/experiment14"
os.makedirs(out_dir, exist_ok=True)
train_loss_path = os.path.join(out_dir, "train_loss.csv")
latest_ckpt_path = os.path.join(out_dir, "latest_ckpt.pth.tar")
best_loss_ckpt_path = os.path.join(out_dir, "best_loss_ckpt.pth.tar")
backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
os.makedirs(backup_ckpts_dir, exist_ok=True)

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
optimizer = AdamW(model.parameters(), lr=lr)
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

print("Searching for checkpoint...")
epoch_start, best_loss = load_or_initialize_training(model, optimizer, latest_ckpt_path, get_best_loss=True)
print("---------------------------------------------------")


data_dir = "/cim/ehoney/ecse626proj/preprocessed_data_64"
train_dataset = ADNIDatasetSlicesInOrder(data_dir, n=slices_at_once, return_target_slice_index=True, symmetric_normalisation=False, data_size=data_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8) # CHANGE to NUM_WORKERS=4

# model, optimizer, train_dataloader, noise_scheduler = accelerator.prepare(
#     model, optimizer, train_dataloader, noise_scheduler
# )

# num_gpus = torch.cuda.device_count()
# print(f"Number of GPUs available: {num_gpus}")

new_best_loss=False
for epoch in range(epoch_start, epochs):
    print(f"Beginning Epoch {epoch + 1}...")

    losses_over_epoch = []

    model.train()
    # s_time = time.time()
    for step, batch in enumerate(train_dataloader):
        # t_time = time.time()
        # print(time.time()-s_time)
        target_slices, condition_slices, first_target_slice_indices = batch

        # print("Slices shapes:", target_slices.shape, condition_slices.shape)

        target_slices = target_slices.to(device)
        condition_slices = condition_slices.to(device)
        first_target_slice_indices = first_target_slice_indices.to(device)

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=target_slices.device).long()
        # print(f"Timesteps shape: {timesteps.shape}")
        noise = torch.randn_like(target_slices, device=target_slices.device, dtype=torch.float32)
        # print(f"Noise dtype: {noise.dtype}; timesteps dtype: {timesteps.dtype}")
        noised_target_slices = noise_scheduler.add_noise(target_slices, noise, timesteps)
        # print(f"noised_target_slices dtype: {noised_target_slices.dtype}")

        # Then I need to mask out noisy images to only include condition slices i.e. non-condition slices should be zero - NOT RIGHT, see below
        # need to noise the target slices and feed in the condition slices as condition, along with the index encodings for the attention layer
        
        timesteps = timesteps.float()

        input_tensor = torch.concat((noised_target_slices, condition_slices), dim=1)

        input_tensor = input_tensor.to(torch.float32)
        timesteps = timesteps.to(torch.float32)
        
        first_target_slice_indices = first_target_slice_indices.unsqueeze(1).unsqueeze(1)
        # print(first_target_slice_indices.shape)
        first_target_slice_indices = first_target_slice_indices.to(torch.float32)

        first_target_slice_indices = first_target_slice_indices / data_size # Normalise slice indices

        # print("Model arg shapes:", input_tensor.shape, timesteps.shape) #, indices_encoding.shape)
        # print("Model arg dtypes:", input_tensor.dtype, timesteps.dtype, indices_encoding.dtype)
        # print(model)
        predicted_noise = model(input_tensor, timesteps, first_target_slice_indices).sample

        # print("Predicted noise", predicted_noise.shape)

        # print("Noise devices", predicted_noise.device, noise.device)
        loss = loss_fn(predicted_noise, noise)
        # print("Loss device", loss.device)

        optimizer.zero_grad()
        loss.backward()
        # accelerator.backward(loss)
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
    if average_epoch_loss < best_loss:
        best_loss = average_epoch_loss
        new_best_loss = True
        print("New best loss!")
        
    print('Saving model checkpoint...')
    checkpoint = {
        'epoch': epoch,
        'model_sd': model.state_dict(),
        'optim_sd': optimizer.state_dict(),
        'loss': average_epoch_loss,
        'best_loss': best_loss
    }
    torch.save(checkpoint, latest_ckpt_path)
    if new_best_loss:
        torch.save(checkpoint, best_loss_ckpt_path)
        new_best_loss=False
    if (epoch + 1) % backup_interval == 0:
        torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch + 1}.pth.tar'))
    print('Checkpoint saved successfully.')