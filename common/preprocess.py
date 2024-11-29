from common.utils import load_from_lz4, normalise_intensity
import os
from skimage.transform import resize
import numpy as np

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data_dir = "/cim/data/adni_class_pred_1x1x1_v1/ADNI"
    out_data_dir = "/cim/ehoney/ecse626proj/preprocessed_data"

    for i, subject in enumerate(os.listdir(data_dir)):
        print(f"Step: {i}. Processing subject {subject}...")
        brain_path = os.path.join(data_dir, subject, 't1p.npy.lz4')
        mask_path = os.path.join(data_dir, subject, 'brainmask.npy.lz4')

        brain_data = load_from_lz4(brain_path)
        mask_data = load_from_lz4(mask_path)
        
        masked_brain_data = mask_data * brain_data
        normalised_brain_data = normalise_intensity(masked_brain_data)
        resized_brain_data = resize(normalised_brain_data, (128, 128, 128), order=3, mode='constant', anti_aliasing=True)

        # print(resized_brain_data.shape)
        # plt.imshow(resized_brain_data[64,], cmap='gray')
        # plt.savefig(f'/cim/ehoney/ecse626proj/{subject}.png')

        np.save(os.path.join(out_data_dir, f"{subject}.npy"), resized_brain_data)
        print("Saved.")