from logging import root
from torch.utils.data import Dataset
import torch
import os
from skimage import io, transform
import numpy as np


class DemosDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 3716

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f'{self.root_dir}/episode_{idx:07d}.npz'
        image = np.load(img_name)['rgb_static']
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample