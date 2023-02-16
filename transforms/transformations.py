from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch
import trimesh
from trimesh import sample
import numpy as np


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


class SamplePoints(BaseTransform):
    def __init__(self, num_points: int, include_normals: bool = False):
        self.num_points = num_points
        self.include_normals = include_normals

    def __call__(self, data: Data) -> Data:
        mesh = trimesh.Trimesh(vertices=data.pos.numpy(), face=data.face.numpy())
        samples, face_indices = sample.sample_surface(mesh, count=self.num_points)

        new_data = Data()

        for key in data.keys:
            if key not in ("pos", "face", "normal"):
                new_data[key] = data[key]

        new_data.pos = torch.from_numpy(samples.astype(np.float32))

        if self.include_normals:
            new_data.normal = torch.from_numpy(mesh.face_normals[face_indices].astype(np.float32))
            new_data.normal /= torch.linalg.norm(new_data.normal, dim=1)

        return new_data


class TestPointSample(BaseTransform):
    def __init__(self, num_points: int = 1024):
        self.point_sampler = SamplePoints(num_points=num_points)

    def __call__(self, data) -> bool:
        self.point_sampler(data)
        return True
