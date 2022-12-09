import pathlib
import json
from typing import Tuple, Any

import umap
from plotly import graph_objects as go
import torch
import trimesh
from torch_geometric.data import Data
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm.auto import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


from point_transformer.models import ClsPointTransformer
from training.data import SimpleShapesDataset


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_network(path_to_config: str, device=None) -> Tuple[ClsPointTransformer, Any, dict, dict, str]:
    config = OmegaConf.load(path_to_config)
    cls_mapping = pathlib.Path(path_to_config).parent / "class_mapping.json"

    with open(cls_mapping, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    pre_transforms = instantiate(config.pre_transform)
    model: ClsPointTransformer = instantiate(config.model)

    if device is None:
        device = get_device()

    state_dict = torch.load(str(pathlib.Path(path_to_config).parent.parent /
                                "checkpoint" / "last.ckpt"))["state_dict"]

    new_state = {}

    for key in tuple(state_dict.keys()):
        new_state[key.replace("model.", "")] = state_dict[key]

    del state_dict
    model.load_state_dict(new_state)
    del new_state
    model.to(device)
    model.eval()

    inv_mapping = {num: label for label, num in class_mapping.items()}

    return model, pre_transforms, class_mapping, inv_mapping, device


def load_val_dataloader(config, batch_size: int = 18):
    data_module: SimpleShapesDataset = instantiate(config.datasets)
    data_module.test_load_sett.batch_size = batch_size
    data_module.setup("validate")
    return data_module.val_dataloader()


def compute_embeddings(model: ClsPointTransformer, val_dataloader, inv_mapping: dict, device):
    all_embeddings = []
    labels = []

    for batch in tqdm(val_dataloader):
        with torch.inference_mode():
            with torch.autocast(device):
                batch = batch.to(device)
                all_embeddings.append(model.get_embedding_data(batch).cpu())
                labels.extend(map(lambda x: inv_mapping[x.item()],
                                  model.predict_class_data(batch).cpu()))

    return torch.concat(all_embeddings).numpy(), labels


def plot_embeddings(embeddings, labels):
    encoder = LabelEncoder()
    num_labels = encoder.fit_transform(labels)
    new_embeddings = umap.UMAP().fit_transform(embeddings)

    fig = plt.Figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    for num in np.unique(num_labels):
        mask = num_labels == num
        points = new_embeddings[mask]
        ax.scatter(x=points[:, 0], y=points[:, 1],
                   label=encoder.inverse_transform([num])[0])

    ax.legend()

    return fig


@torch.inference_mode()
def classify_model(path_to_model, pre_transform_state, model_state: ClsPointTransformer, class_mapping: dict, device: str):
    if not path_to_model:
        return

    model = trimesh.load(path_to_model).dump(concatenate=True)

    if isinstance(model, trimesh.Trimesh):
        data = Data(pos=torch.from_numpy(model.vertices), face=torch.from_numpy(
            model.faces))

    new_data = pre_transform_state(data.to(device))

    class_pred = model_state.predict_class(new_data.x, new_data.pos, torch.zeros(
        (new_data.x.shape[0],), dtype=torch.long, device=device)).item()

    return {index: name for name, index in class_mapping.items()}[class_pred]
