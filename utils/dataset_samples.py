def _plot(ax, image_set, seed, num_rows, num_cols):
    import copy
    import random

    random.seed(seed)

    num_images = len(image_set)
    label_set = set()
    for i in range(num_images):
        _, label = image_set[i]
        if label not in label_set:
            label_set.add(label)

    max_index = num_images - 1
    candidate_labels = set()
    for row in range(num_rows):
        for col in range(num_cols):
            if len(candidate_labels) == 0:
                candidate_labels = copy.deepcopy(label_set)

            while True:
                image, label = image_set[random.randint(0, max_index)]
                if label in candidate_labels:
                    ax[row][col].imshow(image, cmap="gray", aspect="auto")
                    candidate_labels.remove(label)
                    break


def _plot_indices(ax, image_set, indices, num_rows, num_cols):
    i = 0
    for row in range(num_rows):
        for col in range(num_cols):
            image, _ = image_set[indices[i]]
            ax[row][col].imshow(image, cmap="gray", aspect="auto")
            i += 1


def _main():
    import argparse

    import matplotlib.pyplot as plt
    from torchvision.transforms import v2

    import utils.methods as methods

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", help="%(default)s")
    parser.add_argument("--num_rows", type=int, default=3, help="%(default)s")
    parser.add_argument("--num_cols", type=int, default=3, help="%(default)s")
    parser.add_argument(
        "--indices",
        type=str,
        default="0-66-93-121-294-558-772-822-973-1226-1611-1918-2212-2527-2791-3083",
        help="%(default)s",
    )
    parser.add_argument("--seed", type=int, default=42, help="%(default)s")
    args = parser.parse_args()

    # ImageNet indices for chapter1:
    # --indices=1833-22-227-258-265-323-2075-528-547-622-1900-1182-1344-1363-1411-1439-2143-2308-2337-2496-2507-2690-2697-2715

    image_set = methods.get_test_set(args.dataset, v2.ToPILImage())
    if image_set is None:
        _, image_set = methods.get_train_val_sets(
            args.dataset,
            train_ratio=0,
            train_transforms=None,
            val_transforms=v2.ToPILImage(),
            seed=args.seed,
        )

    fig, ax = plt.subplots(
        nrows=args.num_rows,
        ncols=args.num_cols,
        squeeze=False,
    )

    if args.dataset == "imagenet":
        indices = [int(i) for i in args.indices.split("-")]
        num_indices = args.num_rows * args.num_cols

        if len(indices) != num_indices:
            raise Exception(f"The number of indices must be {num_indices}")

        _plot_indices(ax, image_set, indices, args.num_rows, args.num_cols)
    else:
        _plot(ax, image_set, args.seed, args.num_rows, args.num_cols)

    sample_folder = methods.figure_folder / "Dataset Samples"
    if not sample_folder.exists():
        sample_folder.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    figure_name = f"{args.dataset}.pdf"
    fig.savefig(sample_folder / figure_name, format="pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    _main()
