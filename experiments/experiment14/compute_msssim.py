from common.msssim import msssim_real
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_num = 1
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

synthetic_data_dir = '/cim/ehoney/ecse626proj/experiments/experiment14/1500epochs/generated_data'
real_data_dir = '/cim/ehoney/ecse626proj/preprocessed_data_64'
data_dir = synthetic_data_dir

sample_name_list = os.listdir(data_dir)

sample_list = []

for sample_name in sample_name_list[:500]:

    assert sample_name.endswith('.npy')

    sample = np.load(os.path.join(data_dir, sample_name))
    sample_list.append(sample)

# Check list correctly computed
# print(sample_list[40].shape)
# print(len(sample_list))

print(len(sample_list))
avg_real_ssim = msssim_real(sample_list, device=device)