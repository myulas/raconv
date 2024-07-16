def _main():
    import argparse
    import logging

    import matplotlib.pyplot as plt

    from pooling.image_channels import generate_stats, normalize_stats, plot_stats
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
    num_rows = len(pooling_types)

    fig, ax = plt.subplots(
        nrows=num_rows,
        ncols=1,
        figsize=(4.8, 3.6 * num_rows),  # width, height
        sharex=True,
        sharey=True,
    )

    logger.info(f"Plotting has started.")
    for row in range(num_rows):
        num_unique_patches_dict, hw_meter, num_patches_meter = generate_stats(
            image_set,
            channel=-1,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pooling=pooling_types[row],
            scale_factors=scale_factors,
        )
        normalize_stats(num_unique_patches_dict, num_patches_meter)
        plot_stats(ax[row], num_unique_patches_dict)
        logger.info(f"Plot [{row}] has been drawn.")

    pretty_name = methods.get_pretty_dataset_name(args.dataset)
    fig.suptitle(pretty_name)
    fig.supxlabel("Scale Factor")
    fig.supylabel("Percent of Unique Summary Features")

    save_folder = methods.figure_folder / "Image Pooling"
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
