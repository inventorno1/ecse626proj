import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch
from torch.optim import AdamW
from dataset import ADNIDataset
from torch.utils.data import DataLoader
from utils import sample_indices


# from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1500 # should be 1736.1 to match paper
lr = 1e-4
batch_size = 3
num_train_timesteps = 128

model = UNet2DConditionModel(
    sample_size=128,
    in_channels=20, #16
    out_channels=20, #8 #initialise to maximum and compute loss only for indexed slices
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    mid_block_type=None,
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    # not sure about what the lower arguments do
    block_out_channels=(256, 512),
    layers_per_block=2, #default
    cross_attention_dim=128, # ~concat one-hot encodings of condition and target slices~ NO now instead just one 128 vector with labels 0,1,2
    attention_head_dim=8, #default
    norm_num_groups=32, #default
    use_linear_projection=True #DIFFERENT
)

noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
optimizer = AdamW(model.parameters(), lr=lr)
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

data_dir = "/cim/data/adni_class_pred_1x1x1_v1/ADNI"
train_dataset = ADNIDataset(data_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):

        batch = batch.to("cuda:0")
        print(step, batch.shape)

        for i in range(batch_size):
            condition, target, encoding = sample_indices()

        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()
        print(timesteps.shape)
        noise = torch.randn_like(batch).to(device)
        print(noise.shape)
        noisy_images = noise_scheduler.add_noise(batch[target], noise, timesteps)
        # noisy_images[~] = 0

        # Then I need to mask out noisy images to only include condition slices i.e. non-condition slices should be zero - NOT RIGHT, see below

        # need to noise the target slices and feed in the condition slices as condition, along with the index encodings for the attention layer
        

        predicted_noise = model(noisy_images, timesteps, encoding).sample
        break

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()