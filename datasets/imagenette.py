import pathlib
import subprocess

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from datasets.image_dataset import dataset_folder

_imagenette_folder = dataset_folder / "Imagenette"
_imagenette_link = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"


def download():
    archive = pathlib.Path(_imagenette_link).name

    if (_imagenette_folder / archive).exists():
        print("Imagenette - Already Downloaded.")
    else:
        _imagenette_folder.mkdir(exist_ok=True, parents=True)

        cd_cmd = f"cd {_imagenette_folder}"
        download_cmd = f"wget --quiet {_imagenette_link}"
        extract_cmd = f"tar --extract --file {archive} --strip-components=1"

        print("Imagenette - Downloading...")
        subprocess.run(
            args=f"{cd_cmd}&&{download_cmd}&&{extract_cmd}",
            shell=True,
        )
        print("Imagenette - Extracted.")


def get_train_set(transforms):
    download()
    train_set = ImageFolder(_imagenette_folder / "train", transform=transforms)

    return train_set


def get_val_set(transforms):
    download()
    val_set = ImageFolder(_imagenette_folder / "val", transform=transforms)

    return val_set


_mean = [0.4594, 0.4552, 0.4292]
_std = [0.2864, 0.2821, 0.3052]

val_transforms = v2.Compose(
    [
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_mean, std=_std),
    ]
)


def get_train_transform(version):
    transforms = [
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_mean, std=_std),
    ]

    match version:
        case 1:
            return v2.Compose(transforms)
        case 2:
            transforms.append(v2.RandomErasing())

            return v2.Compose(transforms)
        case 3:
            transforms[1] = v2.AutoAugment()

            return v2.Compose(transforms)
        case 4:
            transforms.insert(2, v2.AutoAugment())

            return v2.Compose(transforms)
        case 5:
            transforms.insert(2, v2.AutoAugment())
            transforms.append(v2.RandomErasing())

            return v2.Compose(transforms)
        case 6:
            transforms[1] = v2.RandAugment()

            return v2.Compose(transforms)
        case 7:
            transforms.insert(2, v2.RandAugment())

            return v2.Compose(transforms)
        case 8:
            transforms.insert(2, v2.RandAugment())
            transforms.append(v2.RandomErasing())

            return v2.Compose(transforms)
        case 9:
            transforms[1] = v2.TrivialAugmentWide()

            return v2.Compose(transforms)
        case 10:
            transforms.insert(2, v2.TrivialAugmentWide())

            return v2.Compose(transforms)
        case 11:
            transforms.insert(2, v2.TrivialAugmentWide())
            transforms.append(v2.RandomErasing())

            return v2.Compose(transforms)
        case 12:
            transforms[1] = v2.AugMix()

            return v2.Compose(transforms)
        case 13:
            transforms.insert(2, v2.AugMix())

            return v2.Compose(transforms)
        case 14:
            transforms.insert(2, v2.AugMix())
            transforms.append(v2.RandomErasing())

            return v2.Compose(transforms)
        case default:
            raise Exception("The version is not valid.")
