import torch
import torch.nn.functional as F

_num_unique_patches_dict = {}
_hw_meter = {}
_num_patches_meter = {}


def _pooling_hook(conv_idx, kernel_size, stride, padding, get_summaries, scale_factors):
    scale_factors_dict = {s: float(s) for s in scale_factors}

    def fn(_, input_fmap):
        _hw_meter[conv_idx][0].update(input_fmap[0].size(2))
        _hw_meter[conv_idx][1].update(input_fmap[0].size(3))

        patches = F.unfold(
            input_fmap[0],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ).squeeze(0)
        _num_patches_meter[conv_idx].update(patches.size(1))

        for s in scale_factors:
            summaries = get_summaries(patches, scale_factors_dict[s])
            num_unique_patches = torch.unique(summaries).numel()
            _num_unique_patches_dict[conv_idx][s].update(num_unique_patches)

    return fn


def _add_hooks(features, kernel_size, stride, padding, get_summaries, scale_factors):
    import torchmetrics
    import torch.nn as nn

    conv_layers = filter(lambda l: type(l) is nn.Conv2d, features)

    for conv_idx, conv in enumerate(conv_layers):
        _num_unique_patches_dict[conv_idx] = {
            s: torchmetrics.MeanMetric().cuda() for s in scale_factors
        }
        _hw_meter[conv_idx] = [torchmetrics.MeanMetric().cuda() for _ in range(2)]
        _num_patches_meter[conv_idx] = torchmetrics.MeanMetric().cuda()

        conv.register_forward_pre_hook(
            _pooling_hook(
                conv_idx,
                kernel_size,
                stride,
                padding,
                get_summaries,
                scale_factors,
            )
        )


def _generate_figures(args):
    import matplotlib.pyplot as plt

    from pooling.image_channels import normalize_stats, plot_stats
    from utils.methods import figure_folder

    save_folder = figure_folder / "Feature Map Pooling"
    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    conv_idx = 0
    for group_idx, num_groups in enumerate([6, 6, 4], start=1):
        num_cols = 2
        num_rows = num_groups // num_cols

        fig, ax = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(4.8 * num_cols, 3.6 * num_rows),  # width, height
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for col in range(num_cols):
            for row in range(num_rows):
                normalize_stats(_num_unique_patches_dict[conv_idx], _num_patches_meter[conv_idx])
                plot_stats(ax[row][col], _num_unique_patches_dict[conv_idx])
                ax[row][col].set_title(f"Before Convolution {conv_idx + 1}", loc="right")
                conv_idx += 1

        fig.supxlabel("Scale Factor")
        fig.supylabel("Percent of Unique Summary Features")
        fig.tight_layout()

        figure_name = f"vgg19bn-{args.pooling}{group_idx}.pdf"
        fig.savefig(save_folder / figure_name, format="pdf", dpi=300)
        plt.close(fig)


def _main():
    import argparse

    from torch.utils.data import DataLoader

    from datasets.imagenet import ImageNet, val_transforms
    from models.vgg import VGG
    from pooling.image_channels import get_average_summaries, get_max_summaries
    from tests.validate_vgg import get_weights, perform_inference

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3, help="default:%(default)s")
    parser.add_argument("--S", type=int, default=1, help="default:%(default)s")
    parser.add_argument("--P", type=int, default=0, help="default:%(default)s")
    parser.add_argument("--pooling", type=str, default="avg", help="default:%(default)s")
    parser.add_argument(
        "--scale_factors",
        type=str,
        default="1e6-1e5-1e4-1e3-1e2-1e1-1e0",
        help="default:%(default)s",
    )
    args = parser.parse_args()

    match args.pooling:
        case "avg":
            get_summaries = get_average_summaries
        case "max":
            get_summaries = get_max_summaries
        case default:
            get_summaries = None

    model = VGG(version=19, batch_norm=True, num_classes=1000).cuda()
    model.load_state_dict(get_weights(model.version, model.batch_norm))
    scale_factors = list(map(str.strip, args.scale_factors.split("-")))
    _add_hooks(model.features, args.K, args.S, args.P, get_summaries, scale_factors)

    val_loader = DataLoader(ImageNet(val_transforms), pin_memory=True)
    acc1, acc5 = perform_inference(model, val_loader, num_classes=1000)
    print(f"VGG{model.version}{'BN' if model.batch_norm else ''}")
    print(f"Acc@1: {100 * acc1}%")
    print(f"Acc@5: {100 * acc5}%")

    _generate_figures(args)


if __name__ == "__main__":
    _main()
