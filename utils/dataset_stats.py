def _get_mean_std(image_set):
    import torch

    num_images = len(image_set)
    num_channels = image_set[0][0].size(0)

    num_pixels = 0
    pixel_sum = torch.zeros(num_channels)
    for i in range(num_images):
        image = image_set[i][0].reshape(num_channels, -1)
        num_pixels += image.size(1)
        pixel_sum += image.sum(1)
    mean = (pixel_sum / num_pixels).unsqueeze(1)

    squared_diff = torch.zeros(num_channels)
    for i in range(num_images):
        image = image_set[i][0].reshape(num_channels, -1)
        squared_diff += (image - mean).pow(2).sum(1)
    std = torch.sqrt(squared_diff / (num_pixels - 1))

    return mean.squeeze(1), std


def _main():
    import argparse

    import utils.methods as methods

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", help="%(default)s")
    args = parser.parse_args()

    train_set, _ = methods.get_train_val_sets(
        args.dataset,
        train_ratio=1,
        train_transforms=methods.image_to_tensor_transform,
        val_transforms=None,
        seed=2024,
    )

    pretty_name = methods.get_pretty_dataset_name(args.dataset)
    mean, std = _get_mean_std(train_set)

    print(f"Dataset: {pretty_name} ({len(train_set)} images)")
    print(f"Mean: {mean}")
    print(f"Std : {std}")


if __name__ == "__main__":
    _main()
