import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import datasets.imagenette as imagenette
from models.vgg import VGG
import utils.methods as methods


class WrappedVGG(pl.LightningModule):
    def __init__(
        self,
        version,
        batch_norm,
        learning_rate,
        weight_decay,
        div_factor,
        epochs,
        transform_version,
        seed,
        batch_size,
        num_workers,
        raconv,
        patience,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VGG(version, batch_norm, num_classes=10)
        if raconv is not None:
            self._replace_layers()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=10)

        self.train_loader = self.val_loader = None

    def _replace_layers(self):
        splits = self.hparams.raconv.split(sep="_")
        for split in splits:
            cfg = split.strip().split(sep=",")
            self.model.replace_with_raconv(
                is_cpp=False,
                conv_no=int(cfg[0]),
                summary=cfg[1].strip(),
                scale_factor=float(cfg[2]),
            )

    def training_step(self, batch, batch_idx):
        out = self.forward(batch[0])
        train_loss = F.cross_entropy(out, batch[1])
        self.train_acc.update(out, batch[1])

        self.log_dict(
            {"train_loss": train_loss, "train_acc": self.train_acc},
            on_step=False,
            on_epoch=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch[0])
        val_loss = F.cross_entropy(out, batch[1])
        self.val_acc.update(out, batch[1])

        self.log_dict(
            {"val_loss": val_loss, "val_acc": self.val_acc},
            on_step=False,
            on_epoch=True,
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True,
        )
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate * self.hparams.div_factor,
            div_factor=self.hparams.div_factor,
            epochs=self.hparams.epochs,
            steps_per_epoch=len(self.train_dataloader()),
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def prepare_data(self):
        imagenette.download()

    def train_dataloader(self):
        if self.train_loader is None:
            self.train_loader = DataLoader(
                dataset=imagenette.get_train_set(
                    imagenette.get_train_transform(self.hparams.transform_version),
                ),
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True,
                pin_memory=True,
                generator=torch.Generator().manual_seed(self.hparams.seed),
                worker_init_fn=lambda _: methods.set_seed(torch.initial_seed() % 2**32, False),
                drop_last=True,
            )

        return self.train_loader

    def val_dataloader(self):
        if self.val_loader is None:
            self.val_loader = DataLoader(
                dataset=imagenette.get_val_set(imagenette.val_transforms),
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                pin_memory=True,
                generator=torch.Generator().manual_seed(self.hparams.seed),
                worker_init_fn=lambda _: methods.set_seed(torch.initial_seed() % 2**32, False),
            )

        return self.val_loader


def _main():
    import argparse
    from datetime import timedelta
    import pathlib

    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer

    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.trainer.states import RunningStage

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=11, help="default: %(default)s")
    parser.add_argument(
        "--batch_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="default: %(default)s",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="default: %(default)s")
    parser.add_argument("--weight_decay", type=float, default=0, help="default: %(default)s")
    parser.add_argument("--div_factor", type=float, default=10, help="default: %(default)s")
    parser.add_argument("--epochs", type=int, default=100, help="default: %(default)s")
    parser.add_argument("--transform_version", type=int, default=1, help="default: %(default)s")
    parser.add_argument("--seed", type=int, default=2024, help="default: %(default)s")
    parser.add_argument("--batch_size", type=int, default=16, help="default: %(default)s")
    parser.add_argument("--num_workers", type=int, default=3, help="default: %(default)s")
    parser.add_argument("--raconv", type=str, help="default: %(default)s")

    parser.add_argument("--patience", type=int, help="default: %(default)s")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="default: %(default)s",
    )
    parser.add_argument(
        "--progress_bar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="default: %(default)s",
    )
    parser.add_argument(
        "--save_dir", type=str, default="../Models/Baseline", help="default: %(default)s"
    )
    parser.add_argument("--config_dir", type=str, default="VGG11", help="default: %(default)s")
    parser.add_argument("--version_dir", type=str, help="default: %(default)s")
    parser.add_argument("--ckpt", type=str, help="default: %(default)s")
    args = parser.parse_args()

    methods.set_seed(args.seed)

    csv_logger = CSVLogger(
        save_dir=args.save_dir,
        name=args.config_dir,
        version=args.version_dir,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename="{epoch}--{val_loss:.4f}--{val_acc:.4f}",
        )
    ]

    if args.patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                verbose=True,
                mode="min",
            )
        )

    timer = Timer(verbose=False)
    callbacks.append(timer)

    vgg = WrappedVGG(
        version=args.version,
        batch_norm=args.batch_norm,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        div_factor=args.div_factor,
        epochs=args.epochs,
        transform_version=args.transform_version,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        raconv=args.raconv,
        patience=args.patience,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=csv_logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
        enable_progress_bar=args.progress_bar,
        deterministic=True,
    )
    trainer.fit(vgg, ckpt_path=args.ckpt)

    if args.plot:
        methods.plot_training(pathlib.Path(csv_logger.log_dir))

    time_check = timer.time_elapsed(RunningStage.SANITY_CHECKING)
    time_train = timer.time_elapsed(RunningStage.TRAINING)
    time_validate = timer.time_elapsed(RunningStage.VALIDATING)

    print(f"Total Elapsed Time: {timedelta(seconds=time_check + time_train + time_validate)}")
    print(f"\tSanity Check: {timedelta(seconds=time_check)}")
    print(f"\tTrain: {timedelta(seconds=time_train)}")
    print(f"\tValidate: {timedelta(seconds=time_validate)}")


if __name__ == "__main__":
    _main()
