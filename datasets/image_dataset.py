import pathlib

from torch.utils.data import Dataset

dataset_folder = pathlib.Path.cwd().parent / "Datasets"


class ImageDataset(Dataset):
    def __init__(self, image_set, transforms):
        super().__init__()
        self.image_set = image_set
        self.transforms = transforms

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, index):
        image, label = self.image_set[index]
        image = self.transforms(image)

        return image, label
