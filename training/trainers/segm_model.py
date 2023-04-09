import io
from typing import Dict, Optional

import torch
import torch_scatter
from hydra.utils import instantiate
from matplotlib.cm import get_cmap
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch_geometric.data import Batch, Data
from torchmetrics import JaccardIndex

import wandb
from point_transformer.models.base_model import BaseModel

from ..metrics import AccMean


class SegmTrainer(LightningModule):
    def __init__(self,
                 *,
                 model: BaseModel,
                 cls_mapping: Dict[str, int],
                 optimizer_config: Dict,
                 scheduler_config: Optional[Dict],
                 loss: torch.nn.Module):
        super().__init__()
        self.model = model
        self.cls_mapping = cls_mapping
        self._scheduler_config = scheduler_config
        self._optimizer_config = optimizer_config
        num_classes = len(self.cls_mapping)
        self._train_iou_per_class = JaccardIndex(num_classes, average="none")
        self._train_iou = JaccardIndex(num_classes, average="macro")
        self._val_iou = self._train_iou.clone()
        self._val_iou_per_class = self._train_iou_per_class.clone()
        self._mean_train_loss_per_epoch = AccMean()
        self._mean_val_loss_per_epoch = AccMean()
        self._class_labels = tuple(name for name, _ in sorted(
            self.cls_mapping.items(), key=lambda x: x[1]))
        self._test_stage = "Valid"
        self._train_stage = "Train"
        self._loss = loss
        self._cmap = get_cmap("PiYG")
        self._incorrect_example = None
        self._incorrect_labels_mask = None
        self._ratio_of_incorrect_samples = 0.0

    def configure_optimizers(self):
        optimizer = instantiate(self._optimizer_config, self.model.parameters())

        if self._scheduler_config is not None:
            return [optimizer], [instantiate(self._scheduler_config, optimizer)]

        return optimizer

    def _log_iou_per_class(self, name: str, iou_per_class, batch_size: int):
        self.log(name,
                 {self._class_labels[i]: iou_per_class[i] for i in range(len(self._class_labels))},
                 on_step=False,
                 on_epoch=True,
                 batch_size=batch_size)

    def training_step(self, batch: Batch, batch_idx):
        predicted_logits = self.model.forward_data(batch)
        loss = self._loss(predicted_logits, batch.y)

        with torch.no_grad():
            predicted_class = self.model.predict_class(predicted_logits)

        iou_per_class = self._train_iou_per_class(predicted_class, batch.y)
        moiu = self._train_iou(predicted_class, batch.y)

        self.log(f"{self._train_stage}/mIOU",
                 moiu,
                 on_step=False,
                 on_epoch=True,
                 batch_size=batch.num_graphs)
        self._log_iou_per_class(f"{self._train_stage}/IOU",
                                iou_per_class, batch_size=batch.num_graphs)

        self._mean_train_loss_per_epoch(batch.num_graphs * loss.item(), batch.num_graphs)

        return loss

    def training_epoch_end(self, outputs) -> None:
        train_loss = self._mean_train_loss_per_epoch.compute().cpu().item()
        self.log(f"{self._train_stage}/NLL", train_loss)
        self._mean_train_loss_per_epoch.reset()

    def _log_point_cloud(self, data: Data, incorrect_label_mask: torch.Tensor):

        point_size_config = {
            "material": {
                'cls': 'PointsMaterial',
                'size': 0.025
            }
        }

        if isinstance(self.logger, TensorBoardLogger):
            vertices = data.pos[None, ...]
            colors = torch.tile(torch.tensor(
                [
                    [
                        [255, 0, 0]
                    ]
                ], dtype=torch.uint8), (1, vertices.shape[1], 1))

            colors *= incorrect_label_mask.view(1, -1, 1)

            self.logger.experiment.add_mesh(f"{self._test_stage}/Most_errors_per_point",
                                            vertices=vertices,
                                            colors=colors,
                                            global_step=self.global_step,
                                            config_dict=point_size_config)
        elif isinstance(self.logger, WandbLogger):
            vertices = data.pos

            colors = torch.tile(torch.tensor(
                [
                    [255, 0, 0]
                ], dtype=torch.uint8), (vertices.shape[0], 1))

            colors *= incorrect_label_mask.view(-1, 1)

            self.logger.experiment.log(
                {
                    f"{self._test_stage}/Most_errors_per_point":
                        wandb.Object3D(
                            torch.cat((vertices.detach().cpu(), colors), dim=1).numpy()
                        )
                }
            )

    def validation_step(self, batch: Batch, batch_idx):
        predicted_logits = self.model.forward_data(batch)
        predicted_labels = self.model.predict_class(predicted_logits)

        iou_per_class = self._val_iou_per_class(predicted_labels, batch.y)
        miou = self._val_iou(predicted_labels, batch.y)

        self.log(f"{self._test_stage}/mIOU", miou,
                 on_step=False,
                 on_epoch=True,
                 batch_size=batch.num_graphs)
        self._log_iou_per_class(f"{self._test_stage}/IOU", iou_per_class,
                                batch_size=batch.num_graphs)

        loss = self._loss(predicted_logits, batch.y)
        self._mean_val_loss_per_epoch(batch.num_graphs * loss.item(), batch.num_graphs)

        incorrect_labels_mask = predicted_labels != batch.y
        incorrect_label_ratios = torch_scatter.scatter_sum(incorrect_labels_mask, index=batch.batch).to(torch.float32)
        incorrect_label_ratios /= torch.diff(batch.ptr)

        max_incorrect_labels_index = incorrect_label_ratios.argmax()

        if self._ratio_of_incorrect_samples is None or incorrect_label_ratios[max_incorrect_labels_index] > self._ratio_of_incorrect_samples:
            self._incorrect_labels_mask = incorrect_labels_mask[batch.batch == max_incorrect_labels_index].detach(
            )
            self._incorrect_example = batch.get_example(max_incorrect_labels_index).clone()

    def validation_epoch_end(self, outputs) -> None:
        val_loss = self._mean_val_loss_per_epoch.compute().cpu().item()
        self.log(f"{self._test_stage}/NLL", val_loss)

        self._log_point_cloud(self._incorrect_example, self._incorrect_labels_mask)
        self._mean_val_loss_per_epoch.reset()

        self._incorrect_example = None
        self._incorrect_labels_mask = None
        self._ratio_of_incorrect_samples = 0.0
