from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from datasets.image_dataset import dataset_folder

_imagenet_folder = dataset_folder / "ImageNet"


def _get_val_images():
    val_images_path = _imagenet_folder / "val"

    if val_images_path.exists():
        sorted_val_images = sorted(val_images_path.iterdir())

        return [val_images_path / img for img in sorted_val_images]
    else:
        raise Exception("The folder containing validation images does not exist.")


def _get_val_labels():
    val_labels_path = _imagenet_folder / "val.txt"

    if val_labels_path.exists():
        with open(val_labels_path) as file:
            return [int(line.strip().split()[1]) for line in file.readlines()]
    else:
        raise Exception("The file containing validation labels does not exist.")


val_transforms = v2.Compose(
    [
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageNet(Dataset):
    def __init__(self, val_transforms):
        super().__init__()
        self.val_transforms = val_transforms
        self.val_images = _get_val_images()
        self.val_labels = _get_val_labels()

    def __len__(self):
        return len(self.val_images)

    def __getitem__(self, index):
        image = Image.open(self.val_images[index]).convert("RGB")
        image = self.val_transforms(image)

        label = self.val_labels[index]

        return image, label
