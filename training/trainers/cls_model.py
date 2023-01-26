from typing import Union, Dict, Optional

import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch, Data
from hydra.utils import instantiate
from torchmetrics import ConfusionMatrix, Accuracy
from sklearn.metrics import ConfusionMatrixDisplay
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from point_transformer.models import ClsPointTransformer

from ..metrics import AccMean


class ClsTrainer(LightningModule):
    def __init__(self,
                 *,
                 model: Union[ClsPointTransformer],
                 cls_mapping: Dict[str, int],
                 optimizer_config: Dict,
                 scheduler_config: Optional[Dict]):
        super().__init__()
        self.model = model
        self.cls_mapping = cls_mapping
        self._scheduler_config = scheduler_config
        self._optimizer_config = optimizer_config
        self._conf_matrix = ConfusionMatrix(len(self.cls_mapping), normalize="all")
        self._accuracy = Accuracy(num_classes=len(self.cls_mapping))
        self._mean_loss_per_epoch = AccMean()
        self._class_labels = tuple(name for name, _ in sorted(
            self.cls_mapping.items(), key=lambda x: x[1]))
        self._test_stage = "Test"
        self._train_stage = "Train"

    def configure_optimizers(self):
        optimizer = instantiate(self._optimizer_config, self.model.parameters())

        if self._scheduler_config is not None:
            return [optimizer], [instantiate(self._scheduler_config, optimizer)]

        return optimizer

    def training_step(self, batch: Batch, batch_idx):
        predicted_logits = self.model.forward_data(batch)
        loss = torch.nn.functional.cross_entropy(predicted_logits, batch.y, reduction="mean")
        self._mean_loss_per_epoch(batch.num_graphs * loss.item(), batch.num_graphs)
        return loss

    def training_epoch_end(self, outputs) -> None:
        val_loss = self._mean_loss_per_epoch.compute().cpu().item()
        self.log(f"{self._train_stage}/NLL", val_loss)
        self._mean_loss_per_epoch.reset()

    def _log_point_cloud(self, data: Data, batch_idx):
        point_size_config = {
            "material": {
                'cls': 'PointsMaterial',
                'size': 0.025
            }
        }
        vertices = data.pos[None, ...]
        colors = torch.tile(torch.tensor(
            [
                [
                    [255, 0, 0]
                ]
            ], dtype=torch.uint8), (1, vertices.shape[1], 1))

        if isinstance(self.logger, TensorBoardLogger):
            class_name = self._class_labels[data.y[0]]
            self.logger.experiment.add_mesh(f"{self._test_stage}/{class_name}/{batch_idx}",
                                            vertices=vertices, colors=colors, global_step=self.global_step, config_dict=point_size_config)

    def validation_step(self, batch: Batch, batch_idx):
        predicted_logits = self.model.forward_data(batch)
        self._accuracy(predicted_logits, batch.y)
        self._conf_matrix(predicted_logits, batch.y)
        self.log(f"{self._test_stage}/Accuracy", self._accuracy, on_step=False, on_epoch=True)

        predicted_labels = self.model.predict_class(predicted_logits)

        incorrect_example_index = torch.nonzero(predicted_labels != batch.y).view(-1)

        if len(incorrect_example_index) > 0:
            incorrect_example_index = incorrect_example_index[0].item()
            incorrect_example = batch[incorrect_example_index]
            self._log_point_cloud(incorrect_example, batch_idx)

    def validation_epoch_end(self, outputs) -> None:
        matrix = self._conf_matrix.compute().cpu().numpy()

        conf_plot = ConfusionMatrixDisplay(matrix, display_labels=self._class_labels)
        conf_plot.plot()
        fig = conf_plot.figure_
        fig.set_size_inches(10, 10)

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure(
                f"{self._test_stage}/Conf_matrix", fig, global_step=self.global_step)

        self._conf_matrix.reset()