import os
from skimage.transform import resize
import numpy as np

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data_dir = "/cim/ehoney/ecse626proj/preprocessed_data"
    out_data_dir = "/cim/ehoney/ecse626proj/preprocessed_data_64"
    os.makedirs(out_data_dir, exist_ok=True)

    for i, subject in enumerate(os.listdir(data_dir)):
        print(f"Step: {i}. Processing subject {subject}...")
        subject_path = os.path.join(data_dir, subject)
        brain_data = np.load(subject_path)
        brain_data = resize(brain_data, (64, 64, 64), order=3, mode='constant', anti_aliasing=True)

        # print(resized_brain_data.shape)
        # plt.imshow(resized_brain_data[64,], cmap='gray')
        # plt.savefig(f'/cim/ehoney/ecse626proj/{subject}.png')

        np.save(os.path.join(out_data_dir, f"{subject}"), brain_data)
        print("Saved.")