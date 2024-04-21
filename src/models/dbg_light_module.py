import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
)

from eval.inference_metrics import InferenceMetrics


class DBGLightningModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        criterion: torch.nn.modules.loss._Loss,
        batch_size: int = 1,
        threshold: float = 0.5,
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
        self.test_metrics = InferenceMetrics(threshold=threshold)
        self.cm = BinaryConfusionMatrix(threshold=threshold)

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
        y = batch.data.y

        scores = self.net(x=x.float(), edge_index=edge_index, edge_attr=edge_attr)
        expected_scores = y.float().unsqueeze(-1)
        return scores, expected_scores

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        scores, expected_scores = self.common_step(batch, batch_idx, dataloader_idx)
        loss = self.hparams.criterion(scores, expected_scores)
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
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

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
        scores = torch.sigmoid(scores).reshape((1, -1))
        expected_scores = expected_scores.reshape((1, -1))
        self.test_metrics.update(scores, expected_scores.int(), batch.path)
        self.cm(scores, expected_scores.int())
        # np.save(f"{storage_dir}/results/{batch.path[0].stem}", scores.cpu().numpy())
        # np.save(f"{storage_dir}/results/expected_{batch.path[0].stem}", expected_scores.cpu().numpy())

    def on_test_end(self):
        df = self.test_metrics.finalize()
        print(df.to_string())
        if hasattr(self.logger, "log_table"):
            self.logger.log_table("test/metrics", dataframe=df)
        print(self.cm.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        scores, _ = self.common_step(batch, batch_idx, dataloader_idx)
        scores = torch.sigmoid(scores)
        return torch.round(scores)
