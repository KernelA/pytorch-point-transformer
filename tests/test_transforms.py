import pytest
import torch
import trimesh
from torch_geometric.data import Batch, Data

from transforms.transformations import FPS, SamplePoints


def test_fps_sampling(sample_batch: Batch):
    data = sample_batch.get_example(0)

    downsampled_point_size = data.num_nodes // 2

    fps = FPS(num_points=downsampled_point_size)

    new_data = fps(data)

    assert new_data.num_nodes == pytest.approx(downsampled_point_size, abs=2)
    assert new_data.num_node_features == data.num_node_features
    assert new_data.keys == data.keys


def test_fps_sampling_batch(sample_batch: Batch):
    downsampled_point_size = sample_batch.get_example(0).num_nodes // 2

    fps = FPS(num_points=downsampled_point_size)
    new_data = fps(sample_batch)

    for i in range(sample_batch.num_graphs):
        new_sample = new_data.get_example(i)
        data_sample = sample_batch.get_example(i)

        assert new_sample.num_nodes == pytest.approx(downsampled_point_size, abs=2)
        assert new_sample.num_node_features == data_sample.num_node_features
        assert new_sample.keys == data_sample.keys


def test_normal_transform():
    mesh_data = trimesh.load("./media/chebur.obj")

    data_sample = Data(pos=torch.from_numpy(mesh_data.vertices),
                       face=torch.from_numpy(mesh_data.faces))

    tr = SamplePoints(128, include_normals=True)

    sampled_points = tr(data_sample)

    torch.testing.assert_close(torch.linalg.norm(
        sampled_points.normal, dim=1), torch.full((sampled_points.normal.shape[0], ), 1.0))
