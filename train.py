import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from torch.optim import AdamW
from dataset import ADNIDataset
from torch.utils.data import DataLoader
from utils import sample_16_indices, indices_to_mask, save_tloss_csv, load_or_initialize_training
import numpy as np

# from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1500 # should be 1736.1 to match paper
backup_interval = 5
lr = 1e-4
batch_size = 3
num_train_timesteps = 128
loss_fn = torch.nn.MSELoss()

out_dir = "/cim/ehoney/ecse626proj"
train_loss_path = os.path.join(out_dir, "train_loss.csv")
latest_ckpt_path = os.path.join(out_dir, "latest_ckpt.pth.tar")
backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')

print("---------------------------------------------------")
print("Initialising model...")
model = UNet2DConditionModel(
    sample_size=128,
    in_channels=16, #16
    out_channels=8, #8 #initialise to maximum and compute loss only for indexed slices
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    mid_block_type=None,
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    # not sure about what the lower arguments do
    block_out_channels=(256, 512), #start with 64? no deeper than 1024
    layers_per_block=2, #default
    cross_attention_dim=256, # ~concat one-hot encodings of condition and target slices~ NO now instead just one 128 vector with labels 0,1,2
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

data_dir = "/cim/data/adni_class_pred_1x1x1_v1/ADNI"
train_dataset = ADNIDataset(data_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # CHANGE to NUM_WORKERS=4

for epoch in range(epoch_start, epochs):
    print(f"Beginning Epoch {epoch + 1}...")

    losses_over_epoch = []

    model.train()
    for step, batch in enumerate(train_dataloader):

        batch = batch.to(device)
        # print(step, batch.shape)

        input_tensor = torch.zeros(batch_size, 16, 128, 128, device=device)
        timesteps_tensor = torch.zeros(batch_size, device=device)
        encoding_tensor = torch.zeros(batch_size, 256, device=device)

        noise_tensor = torch.zeros(batch_size, 8, 128, 128, device=device)

        for i, image in enumerate(batch):
            target, condition, encoding = sample_16_indices()

            target_mask = indices_to_mask(target)
            condition_mask = indices_to_mask(condition)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
            prenoise_target = image[target_mask]
            noise = torch.randn_like(prenoise_target, device=device)
            # print(prenoise_target.shape, noise.shape, timesteps.shape)
            noised_target = noise_scheduler.add_noise(prenoise_target, noise, timesteps)

            input = torch.zeros(16, 128, 128, device=device)
            input[:8] = noised_target
            # if conditioning, otherwise will leave bottom 8 input slices as zeros
            if len(condition) > 0:
                input[8:16] = image[condition_mask]

            input_tensor[i] = input
            timesteps_tensor[i] = timesteps
            encoding_tensor[i] = torch.from_numpy(encoding).to(device)

            noise_tensor[i] = noise

        # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()
        # print(timesteps.shape)
        # noise = torch.randn_like(batch).to(device)
        # print(noise.shape)
        # noisy_images = noise_scheduler.add_noise(batch[target], noise, timesteps)
        # noisy_images[~] = 0

        # Then I need to mask out noisy images to only include condition slices i.e. non-condition slices should be zero - NOT RIGHT, see below

        # need to noise the target slices and feed in the condition slices as condition, along with the index encodings for the attention layer
        
        encoding_tensor = encoding_tensor.unsqueeze(1)

        # print(input_tensor.shape, timesteps_tensor.shape, encoding_tensor.shape)
        predicted_noise = model(input_tensor, timesteps_tensor, encoding_tensor).sample
        # print("Predicted noise", predicted_noise.shape)

        # print("Noise devices", predicted_noise.device, noise_tensor.device)
        loss = loss_fn(predicted_noise, noise_tensor)
        # print("Loss device", loss.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")
        losses_over_epoch.append(loss.detach().cpu())
        # break # for debugging, just use one batch

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