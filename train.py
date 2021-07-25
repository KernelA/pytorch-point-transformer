import random
import pathlib
from typing import Dict

import hydra
from hydra.core.config_store import ConfigStore
from torch_geometric import transforms
import numpy as np
import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from data_configs import TrainConfig, PreprocessConfig
from transforms import FusePosAndNormals, FeaturesFromPos

cs = ConfigStore().instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="preprocess", node=PreprocessConfig)


def get_train_transform(num_points: int, include_normals: bool):
    return transforms.Compose([
        transforms.SamplePoints(num=num_points, include_normals=include_normals),
        FeaturesFromPos(),
        FusePosAndNormals()])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_one_epoch(model, optimizer, train_loader, device) -> float:
    model.train()

    epoch_loss = 0

    for data in tqdm.tqdm(train_loader, desc="Train epoch"):
        data = data.to(device)
        optimizer.zero_grad()
        predicted_logits = model(data.x, data.pos, data.batch)
        loss = F.cross_entropy(predicted_logits, data.y, reduction="mean")
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def log_point_cloud(epoch: int, batch_index: int, data, log_writer):
    point_size_config = {
        'material': {
            'cls': 'PointsMaterial',
            'size': 4
        }
    }
    vertices = data.pos[None, ...]
    colors = torch.tile(torch.tensor(
        [[[255, 0, 0]]], dtype=torch.uint8), (1, vertices.shape[1], 1))
    log_writer.add_mesh(f"Test/Incorrect example at {batch_index}",
                        vertices=vertices, colors=colors, global_step=epoch, config_dict={"material": point_size_config})


@ torch.no_grad()
def eval_one_epoch(epoch: int, model, test_loader, log_writer, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    for batch_index, data in enumerate(test_loader):
        true_labels.extend(data.y)
        data = data.to(device)
        predicted = model.predict_class(data.x, data.pos, data.batch)
        incorrect_example_index = torch.nonzero(predicted != data.y).view(-1)
        if len(incorrect_example_index) > 0:
            incorrect_example_index = incorrect_example_index[0].item()
            incorrect_example = data[incorrect_example_index]
            log_point_cloud(epoch, batch_index, incorrect_example, log_writer)
        predicted_labels.extend(predicted.cpu())

    return true_labels, predicted_labels


def plot_conf_matrix(conf_matrix, class_mapping):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Confusion matrix")
    ax.imshow(conf_matrix, interpolation="none", cmap="summer_r")

    labels = tuple(name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1]))
    num_labels = tuple(range(len(labels)))
    ax.set_xticks(num_labels)
    ax.set_xticklabels(labels)
    ax.set_yticks(num_labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in num_labels:
        for j in num_labels:
            text = ax.text(j, i, "{0:.4f}".format(conf_matrix[i, j]),
                           ha="center", va="center", color="black")

    return fig


def train(*, epochs: int, model, optimizer, scheduler, train_loader, test_loader, device, log_dir: pathlib.Path, class_mapping: Dict[str, int]):
    checkpoint_dir = log_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    with SummaryWriter(log_dir=log_dir / "logs", flush_secs=60) as writer:
        progress = tqdm.trange(epochs)

        for epoch in progress:
            epoch_loss = train_one_epoch(model, optimizer, train_loader, device)

            writer.add_scalar("Train/NLL", epoch_loss, global_step=epoch)

            true_labels, predicted_labels = eval_one_epoch(
                epoch, model, test_loader, writer, device)

            if scheduler is not None:
                scheduler.step()

            confusion_matr = confusion_matrix(true_labels, predicted_labels, normalize="all")

            fig = plot_conf_matrix(confusion_matr, class_mapping)
            writer.add_figure("Test/conf_matrix", fig, global_step=epoch)

            overall_acc = np.diag(confusion_matr).sum()
            writer.add_scalar("Test/overall_accuracy", overall_acc, global_step=epoch)

            progress.set_postfix({"Epoch loss": epoch_loss, "Overall test acc": overall_acc})

            del confusion_matr
            torch.save(model.state_dict(), checkpoint_dir / "last.pth")


def enable_cudnn_optimizations():
    torch.backends.cudnn.benchmark = True


@ hydra.main(config_name="train")
def main(config: TrainConfig):
    set_seed(config.seed)

    enable_cudnn_optimizations()

    transform = get_train_transform(
        config.preprocess.num_points, config.preprocess.include_normals)

    dataset = hydra.utils.instantiate(config.datasets,
                                      data_root=config.prepare.data_root,
                                      path_to_zip=config.prepare.path_to_zip,
                                      train_transform=transform,
                                      test_transform=transform,
                                      train_num_workers=config.train_load_workers,
                                      test_num_workers=config.test_load_workers,
                                      train_batch_size=config.train_batch_size,
                                      test_batch_size=config.test_batch_size)

    class_mapping = dataset.get_class_mapping()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(config.model, num_classes=len(class_mapping))
    model.to(device)
    optimizer = hydra.utils.instantiate(config.optimizer, model.parameters())

    if config.scheduler is not None:
        scheduler = hydra.utils.instantiate(config.scheduler, optimizer)
    else:
        scheduler = None

    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()

    exp_dir = pathlib.Path(config.log_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    config_dir = exp_dir / "config"
    config_dir.mkdir(exist_ok=True)

    with open(config_dir / "config.yaml", "w", encoding="utf-8") as file:
        OmegaConf.save(config=config, f=file)

    train(epochs=config.max_epochs, model=model, optimizer=optimizer, scheduler=scheduler,
          train_loader=train_loader, test_loader=test_loader, device=device, log_dir=exp_dir,
          class_mapping=class_mapping)


if __name__ == "__main__":
    main()
