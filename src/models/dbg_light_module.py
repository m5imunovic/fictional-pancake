from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
)


class DBGLightningModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        criterion: torch.nn.modules.loss._Loss,
        batch_size: int = 1,
        threshold: Optional[float] = 0.5,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"], logger=False)
        self.net = net
        self.batch_size = batch_size

        # training metrics
        self.train_acc = BinaryAccuracy()
        # validation metrics
        self.val_acc = BinaryAccuracy()

        # test metrics
        self.test_acc = BinaryAccuracy(threshold=threshold)
        self.binary_precision = BinaryPrecision(threshold=threshold)
        self.recall = BinaryRecall(threshold=threshold)
        self.f1score = BinaryF1Score(threshold=threshold)
        self.cm = BinaryConfusionMatrix(threshold=threshold)
        self.pr_curve = BinaryPrecisionRecallCurve()

        self.scores_all = []
        self.expected_scores_all = []

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
        if isinstance(batch, list):
            batch = batch[0]
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)
        y = batch.y

        scores = self.net(x=x.float(), edge_index=edge_index, edge_attr=edge_attr)
        expected_scores = y.float().unsqueeze(-1)
        return scores, expected_scores

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        loss = self.hparams.criterion(scores, expected_scores)
        self.train_acc(torch.sigmoid(scores), expected_scores.int())

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        loss = self.hparams.criterion(scores, expected_scores)
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
        self.log(
            "val/acc",
            self.val_acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            add_dataloader_idx=True,
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        scores = torch.sigmoid(scores)
        self.recall(scores, expected_scores.int())
        self.binary_precision(scores, expected_scores.int())
        self.test_acc(scores, expected_scores.int())
        self.f1score(scores, expected_scores.int())
        self.cm(scores, expected_scores.int())
        self.pr_curve(scores, expected_scores.int())

        self.scores_all.append(scores.cpu().clone())
        self.expected_scores_all.append(expected_scores.cpu().clone())

    def on_test_end(self):
        if hasattr(self.logger, "log_table"):
            columns = ["accuracy", "precision", "recall", "f1score"]
            data = [
                self.test_acc.compute(),
                self.binary_precision.compute(),
                self.recall.compute(),
                self.f1score.compute(),
            ]
            data = [[entry.item() for entry in data]]
            self.logger.log_table(
                "test/metrics",
                columns=columns,
                data=data,
            )
        print(self.cm.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        scores, _ = self.common_step(batch, batch_idx, dataloader_idx)
        scores = torch.sigmoid(scores)
        return torch.round(scores)
