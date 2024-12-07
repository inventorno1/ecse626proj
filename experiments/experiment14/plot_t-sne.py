import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_num = 1
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def produce_sample_list(sample_name_list, data_dir):
    sample_list = []

    for sample_name in sample_name_list:
        assert sample_name.endswith('.npy')

        sample = np.load(os.path.join(data_dir, sample_name))
        sample_list.append(sample)
    
    return sample_list

def flatten_3d_data(volumes):
    flattened = [vol.flatten() for vol in volumes]
    return np.array(flattened)

def tsne_plot(real_samples, fake_samples, device='cpu'):
    # Flatten the 3D volumes (alternatively, you could use CNN features here)
    real_flattened = flatten_3d_data(real_samples)
    fake_flattened = flatten_3d_data(fake_samples)

    # Combine real and fake samples for PCA and t-SNE (label them for visualization)
    data = np.vstack((real_flattened, fake_flattened))
    labels = np.array([0] * len(real_samples) + [1] * len(fake_samples))  # 0 for real, 1 for fake
    
    # Standardize the data before PCA (important for PCA and t-SNE)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply t-SNE to reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data_scaled)

    return tsne_result, labels

def pca_tsne_plot(real_samples, fake_samples, device='cpu'):
    # Flatten the 3D volumes (alternatively, you could use CNN features here)
    real_flattened = flatten_3d_data(real_samples)
    fake_flattened = flatten_3d_data(fake_samples)

    # Combine real and fake samples for PCA and t-SNE (label them for visualization)
    data = np.vstack((real_flattened, fake_flattened))
    labels = np.array([0] * len(real_samples) + [1] * len(fake_samples))  # 0 for real, 1 for fake
    
    # Standardize the data before PCA (important for PCA and t-SNE)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply PCA to reduce to 50 dimensions
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(data_scaled)

    # Apply t-SNE to reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data_scaled)

    return tsne_result, labels

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    synthetic_data_dir = '/cim/ehoney/ecse626proj/experiments/experiment14/1000epochs/generated_data'
    real_data_dir = '/cim/ehoney/ecse626proj/preprocessed_data_64'

    length = 500

    synthetic_sample_name_list = sorted(os.listdir(synthetic_data_dir))[:length]
    real_sample_name_list = sorted(os.listdir(real_data_dir))[:length]

    synthetic_sample_list = produce_sample_list(synthetic_sample_name_list, synthetic_data_dir)
    real_sample_list = produce_sample_list(real_sample_name_list, real_data_dir)

    print(len(synthetic_sample_list))

    tsne_result, labels = pca_tsne_plot(real_samples=real_sample_list, fake_samples=synthetic_sample_list)

    # Plotting the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], label='Real', alpha=0.7, c='orange', s=30)
    plt.scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], label='Fake', alpha=0.7, c='blue', s=30)
    plt.title(f"t-SNE Visualization of Synthetic and Real MRI data")
    plt.legend()
    plt.savefig(f'/cim/ehoney/ecse626proj/experiments/experiment14/1000_epochs_t-sne.png')
    plt.show()