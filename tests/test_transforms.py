import trimesh
import torch
import numpy as np
from torch_geometric.data import Data

from transforms.transformations import SamplePoints


def test_normal_tranfsorm():
    mesh_data = trimesh.load("./media/chebur.obj")

    data_sample = Data(pos=torch.from_numpy(mesh_data.vertices),
                       face=torch.from_numpy(mesh_data.faces))

    tr = SamplePoints(128, include_normals=True)

    sampled_points = tr(data_sample)

    torch.testing.assert_close(torch.linalg.norm(
        sampled_points.normal, dim=1), torch.full((sampled_points.normal.shape[0], ), 1.0))
