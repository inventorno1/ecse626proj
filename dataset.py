from utils import load_from_lz4, normalise_intensity, sample_16_indices, indices_to_mask
from torch.utils.data import Dataset
import torch
import os
from skimage.transform import resize
import time
import numpy as np

class ADNIDataset(Dataset):

    def __init__(self, data_dir, sample_full_brain=False):
        self.data_dir = data_dir # this is now directory of preprocessed brain MRIs
        self.subject_list = os.listdir(data_dir) # this is now a list of files rather than directories
        self.sample_full_brain = sample_full_brain

    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, idx):

        subject = self.subject_list[idx]
        subject_data_path = os.path.join(self.data_dir, subject)
        assert os.path.isfile(subject_data_path)

        # print(subject_data_path)
        # s1 = time.time()
        resized_brain_data = np.load(subject_data_path)
        # s2 = time.time()
        # print(f"Data loading time: {s2-s1}")
        # print(resized_brain_data.shape, type(resized_brain_data))

        if self.sample_full_brain:
            return resized_brain_data

        # Above is simply loading, normalising and resizing the data
        # Below is sampling specific target and condition slices

        # s5 = time.time()
        target_indices, condition_indices, indices_encoding = sample_16_indices()
        # s6 = time.time()
        # print(f"Sampling time: {s6-s5}")

        # s7 = time.time()
        target_mask = indices_to_mask(target_indices)
        condition_mask = indices_to_mask(condition_indices)

        target_slices = resized_brain_data[target_mask]
        target_slices = torch.from_numpy(target_slices)

        if len(condition_indices) > 0:
            condition_slices = resized_brain_data[condition_mask]
            condition_slices = torch.from_numpy(condition_slices)
        else:
            condition_slices = torch.zeros(8, 128, 128)

        indices_encoding = torch.from_numpy(indices_encoding)

        # s8 = time.time()
        # print(f"Last part: {s8-s7}")

        return target_slices, condition_slices, indices_encoding
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from utils import plot_batch_slices

    data_dir = "/cim/ehoney/ecse626proj/preprocessed_data"
    dataset = ADNIDataset(data_dir)

    batch_size=2

    # # Test initialization and length
    # print(f"Dataset size: {len(dataset)}")

    # # Access a single item
    # sample = dataset[1]
    # print(f"Sample shape: {sample.shape}")

    # plt.imshow(sample[100,], cmap='gray')
    # plt.savefig('/cim/ehoney/ecse626proj/test2.png')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}...")
        # for item in batch:
        #     if item is None:
        #         print("NONE")
        #     else:
        #         print(f"Shape: {item.shape}")
        #         print(f"Type: {item.dtype}")
        # plt.imshow(batch[0, 64,], cmap='gray')
        # plt.savefig(f'/cim/ehoney/ecse626proj/test{i}.png')

        target_slices, condition_slices, indices_encoding = batch
        # print(indices_encoding)

        # plot_batch_slices(target_slices, batch_size=batch_size, save_path=f'/cim/ehoney/ecse626proj/target{i}.png')
        # plot_batch_slices(condition_slices, batch_size=batch_size, save_path=f'/cim/ehoney/ecse626proj/condition{i}.png')

        if i == 2:  # Check the first three batches
            break