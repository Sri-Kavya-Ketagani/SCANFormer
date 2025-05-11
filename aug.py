import h5py
import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torchvision import transforms


# Functions for random rotation and flipping
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# Updated RandomGenerator class
class RandomGenerator(object):
    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Apply random rotation or flip as augmentation
        if self.augment:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)

        sample['image'] = image
        sample['label'] = label
        return sample
