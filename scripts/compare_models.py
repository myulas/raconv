def _get_raconv_indices(baseline_model, raconv_model):
    from layers.raconv import RAConv

    raconv_indices = []
    i = 0
    while True:
        i += 1
        conv_layer = raconv_model.get_conv_layer(i)
        if conv_layer is None:
            break

        if type(conv_layer) is RAConv:
            raconv_model.replace_with_raconv(
                is_cpp=True,
                conv_no=i,
                summary=conv_layer.summary,
                scale_factor=conv_layer.scale_factor,
            )
            baseline_model.replace_with_cpp_conv(i)
            raconv_indices.append(i)

    return raconv_indices


def _main():
    import argparse

    from torch.utils.data import DataLoader

    from datasets.imagenette import val_transforms
    from scripts.train_vgg import WrappedVGG
    from tests.validate_vgg import perform_inference
    import utils.methods as methods

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=str, help="%(default)s")
    parser.add_argument("--raconv_ckpt", type=str, help="%(default)s")
    parser.add_argument("--num_runs", type=int, default=5, help="%(default)s")
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=False, help="%(default)s"
    )
    args = parser.parse_args()

    print(f"Baseline Model: {args.baseline_ckpt}")
    print(f"RAConv Model: {args.raconv_ckpt}")

    baseline_model = WrappedVGG.load_from_checkpoint(args.baseline_ckpt, map_location="cpu").model
    raconv_model = WrappedVGG.load_from_checkpoint(args.raconv_ckpt, map_location="cpu").model

    val_loader = DataLoader(
        dataset=methods.get_train_val_sets(
            "imagenette",
            train_ratio=0,
            train_transforms=None,
            val_transforms=val_transforms,
            seed=2024,
        )[1],
    )

    raconv_indices = _get_raconv_indices(baseline_model, raconv_model)

    for i in range(args.num_runs):
        baseline_acc1, _ = perform_inference(
            baseline_model,
            val_loader,
            num_classes=10,
            progress=args.progress,
        )
        raconv_acc1, _ = perform_inference(
            raconv_model,
            val_loader,
            num_classes=10,
            progress=args.progress,
        )
    print(f"Baseline Acc@1: {100 * baseline_acc1}%")
    print(f"RAConv Acc@1: {100 * raconv_acc1}%")

    for i in raconv_indices:
        conv_layer = baseline_model.get_conv_layer(i)
        raconv_layer = raconv_model.get_conv_layer(i)

        print(f"Conv {i} - {raconv_layer.summary.upper()} {raconv_layer.scale_factor:.2E}")

        conv_time = conv_layer.exec_time.compute().item()
        raconv_time = raconv_layer.exec_time.compute().item()
        time_ratio = round(conv_time / raconv_time, 3)

        print(f"\tConv Time: {round(conv_time * 1000, 3)}ms")
        print(f"\tRAConv Time: {round(raconv_time * 1000, 3)}ms ({time_ratio}x)")

        num_unique_patches = raconv_layer.num_unique_patches.compute().item()
        num_patches = raconv_layer.num_patches
        unique_percentage = round(100 * (num_unique_patches / num_patches), 3)

        print(f"\t#Unique Patches: {round(num_unique_patches, 3)} ({unique_percentage}%)")
        print(f"\t#Patches: {num_patches}")

        print(f"\tMemory Size: {round(raconv_layer.memory_size.compute().item(), 3)}")


if __name__ == "__main__":
    _main()
