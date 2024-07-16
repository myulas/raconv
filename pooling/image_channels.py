import torch
import torchmetrics
import torch.nn.functional as F


def get_average_summaries(patches, scale_factor):
    return patches.mean(0).mul(scale_factor).int()


def get_max_summaries(patches, scale_factor):
    return patches.max(0).values.mul(scale_factor).int()


def generate_stats(image_set, channel, kernel_size, stride, padding, pooling, scale_factors):
    num_unique_patches_dict = {s: torchmetrics.MeanMetric().cuda() for s in scale_factors}
    hw_meter = [torchmetrics.MeanMetric().cuda() for _ in range(2)]
    num_patches_meter = torchmetrics.MeanMetric().cuda()
    scale_factors_dict = {s: float(s) for s in scale_factors}

    get_summaries = None
    match pooling:
        case "avg":
            get_summaries = get_average_summaries
        case "max":
            get_summaries = get_max_summaries
        case default:
            raise Exception("The pooling type is not valid.")

    num_channels = image_set[0][0].size(0)
    channel_patch_size = kernel_size * kernel_size
    if channel == -1:  # all channels
        start_index = 0
        end_index = channel_patch_size * num_channels
    else:
        start_index = channel * channel_patch_size
        end_index = (channel + 1) * channel_patch_size

    num_images = len(image_set)
    for i in range(num_images):
        img = image_set[i][0].unsqueeze(0).cuda()
        hw_meter[0].update(img.size(2))
        hw_meter[1].update(img.size(3))

        patches = F.unfold(
            img,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ).squeeze(0)
        patches = patches[start_index:end_index]
        num_patches_meter.update(patches.size(1))

        for scale_factor in scale_factors:
            summaries = get_summaries(patches, scale_factors_dict[scale_factor])
            num_unique_patches = torch.unique(summaries).numel()
            num_unique_patches_dict[scale_factor].update(num_unique_patches)

    return num_unique_patches_dict, hw_meter, num_patches_meter


def normalize_stats(num_unique_patches_dict, num_patches_meter):
    average_num_patches = num_patches_meter.compute().item()
    for k, v in num_unique_patches_dict.items():
        num_unique_patches_dict[k] = 100 * (v.compute().item() / average_num_patches)


def plot_stats(ax, num_unique_patches_dict):
    x_values = list(num_unique_patches_dict.keys())
    y_values = list(num_unique_patches_dict.values())

    ax.plot(x_values, y_values, color="red", marker="o", linestyle="dashed")
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    for x, y in zip(x_values, y_values):
        ax.text(x, y, f"{round(y, 2)}%", rotation=45, color="black")


def _main():
    import argparse
    import logging

    import matplotlib.pyplot as plt

    import utils.methods as methods

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", help="default:%(default)s")
    parser.add_argument("--kernel_size", type=int, default=3, help="default:%(default)s")
    parser.add_argument("--stride", type=int, default=1, help="default:%(default)s")
    parser.add_argument("--padding", type=int, default=0, help="default:%(default)s")
    parser.add_argument(
        "--scale_factors",
        type=str,
        default="1e6-1e5-1e4-1e3-1e2-1e1-1e0",
        help="default:%(default)s",
    )
    parser.add_argument(
        "--pooling_types",
        type=str,
        default="avg-max",
        help="default:%(default)s",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    image_set = methods.get_test_set(args.dataset, methods.image_to_tensor_transform)
    if image_set is None:
        _, image_set = methods.get_train_val_sets(
            args.dataset,
            train_ratio=0,
            train_transforms=None,
            val_transforms=methods.image_to_tensor_transform,
            seed=2024,
        )

    scale_factors = [x.strip() for x in args.scale_factors.split("-")]
    pooling_types = [x.strip() for x in args.pooling_types.split("-")]

    num_channels = image_set[0][0].size(0)
    num_rows = len(pooling_types) if num_channels == 1 else num_channels
    num_cols = 1 if num_channels == 1 else len(pooling_types)

    fig, ax = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(4.8 * num_cols, 3.6 * num_rows),  # width, height
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    logger.info(f"Plotting has started.")
    for row in range(num_rows):
        for col in range(num_cols):
            num_unique_patches_dict, hw_meter, num_patches_meter = generate_stats(
                image_set,
                channel=-1 if num_channels == 1 else row,
                kernel_size=args.kernel_size,
                stride=args.stride,
                padding=args.padding,
                pooling=pooling_types[row if num_channels == 1 else col],
                scale_factors=scale_factors,
            )
            normalize_stats(num_unique_patches_dict, num_patches_meter)
            plot_stats(ax[row][col], num_unique_patches_dict)
            logger.info(f"Plot [{row}][{col}] has been drawn.")

    pretty_name = methods.get_pretty_dataset_name(args.dataset)
    fig.suptitle(pretty_name)
    fig.supxlabel("Scale Factor")
    fig.supylabel("Percent of Unique Summary Features")

    save_folder = methods.figure_folder / "Image Channel Pooling"
    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    figure_name = f"{args.dataset}.pdf"
    fig.savefig(save_folder / figure_name, format="pdf", dpi=300)
    plt.close(fig)

    average_num_patches = round(num_patches_meter.compute().item(), 2)
    average_height = round(hw_meter[0].compute().item(), 2)
    average_width = round(hw_meter[1].compute().item(), 2)

    print(f"Dataset: {pretty_name} ({len(image_set)} images)")
    print(f"Average Height of an Image: {average_height}")
    print(f"Average Width of an Image : {average_width}")
    print(f"Average Number of Patches in an Image: {average_num_patches}")


if __name__ == "__main__":
    _main()
