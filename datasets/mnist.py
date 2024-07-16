import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from datasets.image_dataset import dataset_folder, ImageDataset


def get_train_val_sets(train_ratio, train_transforms, val_transforms, seed):
    dataset = MNIST(
        root=dataset_folder,
        train=True,
        transform=None,
        download=True,
    )
    num_images = len(dataset)
    num_train_images = int(train_ratio * num_images)
    num_val_images = num_images - num_train_images

    train_split, val_split = random_split(
        dataset,
        lengths=[num_train_images, num_val_images],
        generator=torch.Generator().manual_seed(seed),
    )
    train_set = ImageDataset(train_split, train_transforms)
    val_set = ImageDataset(val_split, val_transforms)

    return train_set, val_set


def get_test_set(transforms):
    test_set = MNIST(
        root=dataset_folder,
        train=False,
        transform=transforms,
        download=True,
    )

    return test_set
