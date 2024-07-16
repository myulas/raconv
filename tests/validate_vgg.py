import torch
from torchmetrics.classification import Accuracy
import torchvision.models.vgg as vgg
from tqdm import tqdm


@torch.inference_mode()
def perform_inference(model, val_loader, num_classes, progress=True):
    device = next(model.parameters()).device
    acc1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    acc5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    model.eval()
    for images, labels in tqdm(val_loader, disable=not progress):
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)

        acc1.update(out, labels)
        acc5.update(out, labels)

    return acc1.compute(), acc5.compute()


def get_weights(version, batch_norm):
    if batch_norm:
        match version:
            case 11:
                return vgg.VGG11_BN_Weights.DEFAULT.get_state_dict()
            case 13:
                return vgg.VGG13_BN_Weights.DEFAULT.get_state_dict()
            case 16:
                return vgg.VGG16_BN_Weights.DEFAULT.get_state_dict()
            case 19:
                return vgg.VGG19_BN_Weights.DEFAULT.get_state_dict()
            case default:
                raise Exception("The version is not valid.")
    else:
        match version:
            case 11:
                return vgg.VGG11_Weights.DEFAULT.get_state_dict()
            case 13:
                return vgg.VGG13_Weights.DEFAULT.get_state_dict()
            case 16:
                return vgg.VGG16_Weights.DEFAULT.get_state_dict()
            case 19:
                return vgg.VGG19_Weights.DEFAULT.get_state_dict()
            case default:
                raise Exception("The version is not valid.")


if __name__ == "__main__":
    import argparse

    from torch.utils.data import DataLoader

    from datasets.imagenet import val_transforms, ImageNet
    from models.vgg import VGG

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=16, help="default: %(default)s")
    parser.add_argument(
        "--batch_norm",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="default: %(default)s",
    )
    args = parser.parse_args()

    val_loader = DataLoader(
        ImageNet(val_transforms),
        batch_size=64,
        num_workers=3,
        shuffle=False,
        pin_memory=True,
    )

    model = VGG(args.version, args.batch_norm, num_classes=1000)
    model.load_state_dict(get_weights(model.version, model.batch_norm))
    model.cuda()

    acc1, acc5 = perform_inference(model, val_loader, num_classes=1000)
    print(f"VGG{model.version}{'BN' if model.batch_norm else ''}")
    print(f"Acc@1: {100 * acc1}%")
    print(f"Acc@5: {100 * acc5}%")
