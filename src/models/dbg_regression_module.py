from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.regression import MeanSquaredError

from eval.inference_metrics import InferenceMetrics
from models.loss.mixture_loss import MixtureLoss
from utils.container import Container


class DBGRegressionModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        criterion: torch.nn.modules.loss._Loss,
        batch_size: int = 1,
        threshold: int = 0.5,  # in regression setting we use this to adapt range for softmax operation
        storage_path: Path | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"], logger=False)
        self.net = net
        self.batch_size = batch_size

        # training metrics
        self.train_metric = MeanSquaredError()
        # validation metrics
        self.val_metric = MeanSquaredError()

        # test metrics
        self.test_metrics = InferenceMetrics(threshold=0.5)
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
        device = edge_index.device
        edge_attr = getattr(batch.data, "edge_attr", None)
        graph_attr = getattr(batch.data, "graph_attr", None)
        ei_ptr = torch.tensor(batch.ei_ptr, device=device)

        scores = self.net(x=x.float(), edge_index=edge_index, edge_attr=edge_attr, graph_attr=graph_attr, ei_ptr=ei_ptr)
        # scores = torch.clamp(scores, min=0)

        if getattr(batch.data, "y") is not None:
            y = batch.data.y
            expected_scores = y.float().unsqueeze(-1)
        else:
            expected_scores = None
        return scores, expected_scores

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)

        if self.uses_mixture_loss():
            loss = self.hparams.criterion(batch.data.edge_index, scores, expected_scores)
        else:
            weights = None
            if getattr(batch.data, "weights", None) is not None:
                weights = batch.data.weights.unsqueeze(-1)
            if weights is not None:
                loss = self.hparams.criterion(scores, expected_scores, weights)
            else:
                loss = self.hparams.criterion(scores, expected_scores)
        self.train_metric(scores, expected_scores.int())

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
        output = self.train_metric.compute()
        self.log(
            "train/mse",
            output,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        if self.uses_mixture_loss():
            loss = self.hparams.criterion(batch.data.edge_index, scores, expected_scores)
        else:
            weights = None
            if getattr(batch.data, "weights", None) is not None:
                weights = batch.data.weights.unsqueeze(-1)
            if weights is not None:
                loss = self.hparams.criterion(scores, expected_scores, weights)
            else:
                loss = self.hparams.criterion(scores, expected_scores)
        self.val_metric.update(scores, expected_scores.int())

        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def on_validation_epoch_end(self):
        output = self.val_metric.compute()
        self.log(
            "val/mse",
            output,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.val_metric.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        scores = scores.reshape((1, -1))
        expected_scores = expected_scores.reshape((1, -1))
        if self.storage_path is not None and self.batch_size == 1:
            torch.save(scores.cpu(), f"{self.storage_path/batch.path[0].stem}.pt")
            np.save(f"{self.storage_path/batch.path[0].stem}", scores.cpu().numpy())
            np.save(f"{self.storage_path}/expected_{batch.path[0].stem}", expected_scores.cpu().numpy())
            torch.save(batch.data, f"{self.storage_path}/transformed_{batch.path[0].stem}.pt")

        # We define a hyperparameter threshold (offset) which shifts the range to negative values
        # After that we will clamp so everything negative becomes "falsy"
        # This way we can still use metrics as for classification case
        scores = scores - self.hparams.threshold
        scores_bin = torch.clamp(scores, 0).bool().long()
        expected_scores_bin = expected_scores.bool().long()

        self.test_metrics.update(scores_bin, expected_scores_bin, batch.path)

    def on_test_end(self):
        df = self.test_metrics.finalize()
        print(df.to_string())
        if hasattr(self.logger, "log_table"):
            self.logger.log_table("test/metrics", dataframe=df)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        scores, _ = self.common_step(batch, batch_idx, dataloader_idx)
        scores = scores - self.hparams.threshold
        scores = torch.clamp(scores, min=0)
        if self.storage_path is not None:
            if self.batch_size == 1:
                torch.save(scores.cpu(), f"{self.storage_path/batch.path[0].stem}.pt")
            container = Container({"multiplicity": scores.cpu().to(torch.float32)})
            container.save(self.storage_path)
        return scores

    def uses_mixture_loss(self):
        return isinstance(self.hparams.criterion, MixtureLoss)
