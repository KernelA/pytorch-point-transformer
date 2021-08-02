from data.mesh_data import BatchedData
import torch
import trimesh
from trimesh import sample
import numpy as np


class FeaturesFromPos:
    def __call__(self, data):
        data.x = data.pos.clone()
        return data


class FusePosAndNormals:
    def __call__(self, data):
        data.x = torch.cat((data.x, data.normal), dim=-1)
        del data.normal
        return data


class SamplePoints:
    def __init__(self, num_points: int, include_normals: bool = False):
        self.num_points = num_points
        self.include_normals = include_normals

    def __call__(self, data):
        mesh = trimesh.Trimesh(vertices=data.pos.numpy(), faces=data.face.T.numpy())
        samples, face_indices = sample.sample_surface(mesh, count=self.num_points)
        new_data = BatchedData(x=data.x)

        for key in data.keys:
            if key not in ("pos", "face", "normal"):
                new_data[key] = data[key]

        new_data.pos = torch.from_numpy(samples.astype(np.float32))

        if self.include_normals:
            new_data.normal = torch.from_numpy(mesh.face_normals[face_indices].astype(np.float32))

        return new_data


class TestPointSample:
    def __init__(self, num_points: int = 1024):
        self.point_sampler = SamplePoints(num_points=num_points)

    def __call__(self, data) -> bool:
        self.point_sampler(data)
        return True
