import os
import h5py
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import numpy as np

# Dataset class
class ACDCdataset(Dataset):
    def __init__(self, base_dir, list_dir, split, output_size, transform=None):
        self.transform = transform
        self.split = split
        self.output_size = output_size
        self.sample_list = open(os.path.join(list_dir, self.split + '.list')).readlines()
        self.data_dir = os.path.join(base_dir, "data", "slices")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Get the slice name from the list
        slice_name = self.sample_list[idx].strip('\n')  # Remove any newline characters
        data_path = os.path.join(self.data_dir, slice_name + '.h5')  # Add `.h5` extension
        with h5py.File(data_path, 'r') as h5_file:
            image = h5_file['image'][:]  # Assuming dataset contains 'image' key
            label = h5_file['label'][:]  # Assuming dataset contains 'label' key

        #print(f"Image shape: {image.shape}, Label shape: {label.shape}")

        # Resize image and label to the desired output size
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # Resize image
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # Resize label

        # Convert numpy arrays to torch tensors
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label.astype(np.float32))

        # Prepare sample dictionary
        sample = {'image': image, 'label': label.long()}
        
        # Apply transformations if provided and in training phase
        if self.transform and self.split == "train":
            sample = self.transform(sample)
        
        sample['case_name'] = slice_name
        return sample