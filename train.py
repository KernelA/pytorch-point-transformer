import random
import os

import hydra
from hydra.core.config_store import ConfigStore
from torch_geometric import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from data_configs import TrainConfig, DataConfig
from data import SimpleShapesDataset

cs = ConfigStore().instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="data", node=DataConfig)


def get_train_transform(num_points: int):
    return transforms.Compose([
        transforms.SamplePoints(num=num_points),
        transforms.NormalizeScale()]
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_one_epoch(model, optimizer, train_loader, device) -> float:
    model.train()

    epoch_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        predicted_logits = model(data.x, data.pos, data.batch)
        loss = F.cross_entropy(predicted_logits, data.y, reduction="mean")
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


@torch.no_grad()
def eval_one_epoch(model, test_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    for data in test_loader:
        true_labels.extend(data.y)
        data = data.to(device)
        predicted = model.predict_class(data.x, data.pos, data.batch).cpu()
        predicted_labels.extend(predicted)

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


def train(*, epochs: int, model, optimizer, scheduler, train_loader, test_loader, device, log_dir, class_mapping):
    with SummaryWriter(log_dir=os.path.join(log_dir, "logs"), flush_secs=60) as writer:
        progress = tqdm.trange(epochs)

        for epoch in progress:
            epoch_loss = train_one_epoch(model, optimizer, train_loader, device)

            writer.add_scalar("Train/NLL", epoch_loss, global_step=epoch)

            true_labels, predicted_labels = eval_one_epoch(model, test_loader, device)

            if scheduler is not None:
                scheduler.step()

            confusion_matr = confusion_matrix(true_labels, predicted_labels, normalize="all")

            fig = plot_conf_matrix(confusion_matr, class_mapping)
            writer.add_figure("Test/conf_matrix", fig, global_step=epoch)

            overall_acc = np.diag(confusion_matr).sum()
            writer.add_scalar("Test/overall_accuracy", overall_acc, global_step=epoch)

            progress.set_postfix({"Epoch loss": epoch_loss, "Overall test acc": overall_acc})

            del confusion_matr


@ hydra.main(config_name="train")
def main(config: TrainConfig):
    set_seed(config.seed)

    pre_transform = get_train_transform(config.data.num_points)

    dataset = SimpleShapesDataset(data_root=config.data.data_root,
                                  path_to_zip=config.data.path_to_zip,
                                  train_pre_transform=pre_transform,
                                  test_pre_transform=pre_transform,
                                  train_num_workers=config.train_load_workers,
                                  test_num_workers=config.test_load_workers,
                                  train_batch_size=config.train_batch_size,
                                  test_batch_size=config.test_batch_size)

    class_mapping = dataset.get_class_mapping()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(config.model, num_classes=len(class_mapping))
    model.to(device)
    optimizer = hydra.utils.instantiate(config.optimizer, model.parameters())

    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()

    os.makedirs(config.log_dir, exist_ok=True)

    train(epochs=config.max_epochs, model=model, optimizer=optimizer, scheduler=None,
          train_loader=train_loader, test_loader=test_loader, device=device, log_dir=config.log_dir,
          class_mapping=class_mapping)


if __name__ == "__main__":
    main()
