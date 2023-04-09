from typing import Union

import numpy as np
import torch
import trimesh
from torch_cluster import fps
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from trimesh import sample


class FeaturesFromPos(BaseTransform):
    def __call__(self, data: Data) -> Data:
        if "x" in data:
            data.x = torch.cat((data.x, data.pos.clone()), dim=-1)
        else:
            data.x = data.pos.clone()
        return data


class PosToFloat32(BaseTransform):
    def __call__(self, data: Data) -> Data:
        data.pos = data.pos.to(torch.float32)
        return data


class FeaturesFromNormal(BaseTransform):
    def __call__(self, data: Data) -> Data:
        if "x" in data:
            data.x = torch.cat((data.x, data.normal), dim=-1)
        else:
            data.x = data.normal.clone()
        del data.normal
        return data


class FPS(BaseTransform):
    def __init__(self, num_points: int, device: str = "cpu") -> None:
        super().__init__()
        self.num_points = num_points
        self.device = device

    def _copy_node_data(self, data: Data, index):
        new_data = Data()

        for key in data.keys:
            if data[key].numel() > 1:
                new_data[key] = data[key][index]
            else:
                new_data[key] = data[key]

        return new_data

    def __call__(self, data: Union[Data, Batch]) -> Data:
        num_nodes = data.get_example(0).num_nodes if isinstance(data, Batch) else data.num_nodes

        assert num_nodes != 0 and num_nodes is not None

        if isinstance(data, Batch):
            assert len(set(data.get_example(i).num_nodes for i in range(data.num_graphs))
                       ) == 1, "Number of nodes must be equal between all examples"

        ratio = self.num_points / num_nodes

        batch = None

        if isinstance(data, Batch):
            batch = data.batch.to(self.device)

        index = fps(data.pos.to(self.device), ratio=ratio, batch=batch, random_start=False).cpu()

        if isinstance(data, Batch):
            new_batch = batch[index]
            new_data = Batch.from_data_list(
                [
                    self._copy_node_data(data.get_example(example_index), index[new_batch == example_index] - data.ptr[example_index]) for example_index in range(data.num_graphs)
                ]
            )
        else:
            new_data = Data()
            new_data = self._copy_node_data(data, index)

        return new_data


class SamplePoints(BaseTransform):
    def __init__(self, num_points: int, include_normals: bool = False):
        self.num_points = num_points
        self.include_normals = include_normals

    def __call__(self, data: Data) -> Data:
        if "y" in data:
            if data.y.numel() > 1:
                raise RuntimeError("Cannot apply sampling on data with multiple values of 'y'")

        mesh = trimesh.Trimesh(vertices=data.pos.numpy(), faces=data.face.numpy())
        samples, face_indices = sample.sample_surface(mesh, count=self.num_points)

        new_data = Data()

        for key in data.keys:
            if key not in ("pos", "face", "normal"):
                new_data[key] = data[key]

        new_data.pos = torch.from_numpy(samples.astype(np.float32))

        if self.include_normals:
            new_data.normal = torch.from_numpy(mesh.face_normals[face_indices].astype(np.float32))
            new_data.normal /= torch.linalg.norm(new_data.normal, dim=1, keepdim=True)

        return new_data
