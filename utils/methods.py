import os
import pathlib
import random

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

import datasets.cifar10 as cifar10
import datasets.imagenet as imagenet
import datasets.imagenette as imagenette
import datasets.mnist as mnist

figure_folder = pathlib.Path.cwd().parent / "Figures"

image_to_tensor_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def get_pretty_dataset_name(dataset_name):
    match dataset_name:
        case "cifar10":
            return "CIFAR-10"
        case "imagenet":
            return "ImageNet"
        case "imagenette":
            return "Imagenette"
        case "mnist":
            return "MNIST"
        case default:
            raise Exception("The dataset name is not valid.")


def get_train_val_sets(dataset_name, train_ratio, train_transforms, val_transforms, seed):
    match dataset_name:
        case "cifar10":
            train_set, val_set = cifar10.get_train_val_sets(
                train_ratio=train_ratio,
                train_transforms=train_transforms,
                val_transforms=val_transforms,
                seed=seed,
            )

            return train_set, val_set
        case "imagenet":
            return None, imagenet.ImageNet(val_transforms)
        case "imagenette":
            train_set = imagenette.get_train_set(train_transforms)
            val_set = imagenette.get_val_set(val_transforms)

            return train_set, val_set
        case "mnist":
            train_set, val_set = mnist.get_train_val_sets(
                train_ratio=train_ratio,
                train_transforms=train_transforms,
                val_transforms=val_transforms,
                seed=seed,
            )

            return train_set, val_set
        case default:
            raise Exception("The dataset name is not valid.")


def get_test_set(dataset_name, transforms):
    match dataset_name:
        case "cifar10":
            return cifar10.get_test_set(transforms)
        case "imagenet":
            return None
        case "imagenette":
            return None
        case "mnist":
            return mnist.get_test_set(transforms)
        case default:
            raise Exception("The dataset name is not valid.")


def set_seed(seed, set_cupy=True):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)

    if set_cupy:
        cp.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def plot_training(log_dir):
    num_rows = 2
    num_cols = 1

    fig, ax = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(4.8 * num_cols, 3.6 * num_rows),  # width, height
        sharex=True,
    )

    with open(log_dir / "metrics.csv", mode="r") as file:
        lines = file.readlines()

    value_dict = {k: [] for k in lines[0].strip().split(sep=",")}
    dict_keys = list(value_dict.keys())

    for line in lines[1:]:
        values = line.strip().split(sep=",")
        for k, v in zip(dict_keys, values):
            if (k == "epoch" or k == "step") or v == "":
                continue
            value_dict[k].append(float(v))

    x_axis = range(1, len(np.array(value_dict["train_loss"])) + 1)

    ax[0].plot(x_axis, np.array(value_dict["train_loss"]), label="train")
    ax[0].plot(x_axis, np.array(value_dict["val_loss"]), label="val")
    ax[0].set_ylabel(ylabel="Loss")

    ax[0].set_xlim(left=0)
    ax[0].set_ylim(bottom=0)
    ax[0].legend()

    ax[1].plot(x_axis, np.array(value_dict["train_acc"]) * 100, label="train")
    ax[1].plot(x_axis, np.array(value_dict["val_acc"]) * 100, label="val")
    ax[1].set_ylabel(ylabel="Accuracy (%)")

    ax[1].set_xlim(left=0)
    ax[1].set_ylim(bottom=0)
    ax[1].legend()

    fig.supxlabel("Epoch")
    fig.tight_layout()
    fig.savefig(log_dir / "loss_accuracy.pdf", format="pdf", dpi=300)
    plt.close(fig)
