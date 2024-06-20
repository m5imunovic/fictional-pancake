from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
)
from torchmetrics.regression import MeanSquaredError

from eval.inference_metrics import InferenceMetrics
from models.loss.rmse_loss import RMSELoss


class DBGLightningModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        criterion: torch.nn.modules.loss._Loss,
        batch_size: int = 1,
        threshold: float = 0.5,
        offset: int = 0.5,  # in regression setting we use this to adapt range for softmax operation
        storage_path: Path | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"], logger=False)
        self.net = net
        self.batch_size = batch_size

        if self.regression_mode():
            # training metrics
            self.train_acc = MeanSquaredError()
            # validation metrics
            self.val_acc = MeanSquaredError()
        else:
            # training metrics
            self.train_acc = BinaryAccuracy()
            # validation metrics
            self.val_acc = BinaryAccuracy()

        # test metrics
        self.test_metrics = InferenceMetrics(threshold=threshold)
        self.cm = BinaryConfusionMatrix(threshold=threshold)
        self.storage_path = None
        if storage_path is not None:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(exist_ok=True, parents=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def common_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.data.x
        edge_index = batch.data.edge_index
        edge_attr = getattr(batch.data, "edge_attr", None)

        scores = self.net(x=x.float(), edge_index=edge_index, edge_attr=edge_attr)
        if self.regression_mode():
            scores = torch.clamp(scores, min=0)

        if getattr(batch.data, "y") is not None:
            y = batch.data.y
            expected_scores = y.float().unsqueeze(-1)
        else:
            expected_scores = None
        return scores, expected_scores

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        loss = self.hparams.criterion(scores, expected_scores)
        if self.regression_mode():
            self.train_acc(scores, expected_scores.int())
        else:
            self.train_acc(torch.sigmoid(scores), expected_scores.int())

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        output = self.train_acc.compute()
        self.log(
            "train/acc",
            output,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        loss = self.hparams.criterion(scores, expected_scores)
        if self.regression_mode():
            self.val_acc.update(scores, expected_scores.int())
        else:
            self.val_acc.update(torch.sigmoid(scores), expected_scores.int())

        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            add_dataloader_idx=True,
        )
        return loss

    def on_validation_epoch_end(self):
        output = self.val_acc.compute()
        self.log(
            "val/acc",
            output,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            add_dataloader_idx=True,
        )
        self.val_acc.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        if self.regression_mode():
            # We define a hyperparameter offset which shifts the range to negative values
            # After that we will normalize with sigmoid so everything negative becomes "falsy"
            # This way we can still use metrics as for classification case
            scores = scores - self.hparams.offset
        else:
            scores = torch.sigmoid(scores)
        scores = scores.reshape((1, -1))
        expected_scores = expected_scores.reshape((1, -1))
        if self.storage_path is not None:
            np.save(f"{self.storage_path}/{batch.path[0].stem}", scores.cpu().numpy())
            np.save(f"{self.storage_path}/expected_{batch.path[0].stem}", expected_scores.cpu().numpy())
            torch.save(batch.data, f"{self.storage_path}/transformed_{batch.path[0].stem}.pt")
        self.test_metrics.update(scores, expected_scores.int(), batch.path)
        self.cm(scores, expected_scores.int())

    def on_test_end(self):
        df = self.test_metrics.finalize()
        print(df.to_string())
        if hasattr(self.logger, "log_table"):
            self.logger.log_table("test/metrics", dataframe=df)
        print(self.cm.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        scores, _ = self.common_step(batch, batch_idx, dataloader_idx)
        if self.regression_mode():
            # We define a hyperparameter offset which shifts the range to negative values
            # After that we will normalize with sigmoid so everything negative becomes "falsy"
            # This way we can still use metrics as for classification case
            scores = scores - self.hparams.offset
            scores = torch.clamp(scores, min=0)
        else:
            scores = torch.sigmoid(scores).reshape((1, -1))
        if self.storage_path is not None:
            torch.save(scores.cpu(), f"{self.storage_path}/{batch.path[0].stem}.pt")
        return scores

    def regression_mode(self):
        return isinstance(self.hparams.criterion, RMSELoss) or isinstance(self.hparams.criterion, nn.PoissonNLLLoss)
