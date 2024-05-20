"""Implementation of lightning module"""
import torch.linalg
from lightning import pytorch as pl
import torchmetrics

from models.model_selection import ModelSelection


# pylint: disable=W0221, R0913, R0902
class Model(pl.LightningModule):
    """Class implementation of lightning module"""
    def __init__(self,
                 lr: float,
                 lr_patience: int,
                 lr_factor: float,
                 n_classes: int,
                 model_selection: str = "resnet18"):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.loss_function = torch.nn.CrossEntropyLoss()

        match model_selection:
            case "resnet18":
                self.network = ModelSelection.resnet18_custom_model(n_classes)
            case "resnet50":
                self.network = ModelSelection.resnet50_torchvision(n_classes, pretrained=True)
            case _:
                self.network = ModelSelection.resnet18_custom_model(n_classes)

        assert self.network is not None, "Selected network is not valid therefore training is aborted"

        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanAbsolutePercentageError(),
            torchmetrics.MeanSquaredError(),
        ])

        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.forward(x)
        loss = self.loss_function(y_pred, y)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, _batch_idx) -> None:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.val_metrics.update(outputs, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics)

    def test_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.test_metrics.update(outputs, labels)

        self.log('test_loss', loss, prog_bar=True)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            betas=(0.91, 0.9999),
            lr=self.lr,
            weight_decay=0.1,
            amsgrad=True
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.lr_patience,
            factor=self.lr_factor
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_MeanAbsoluteError',
        }
