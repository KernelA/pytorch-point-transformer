from data.mesh_data import BatchedData
import torch
import pytest

from torch_geometric import data as tg_data

from data import BatchedData

NUM_COORDS = 3
NUM_POINTS = 10
SEED = 20


@pytest.fixture()
def sample_batch():
    generator = torch.Generator()
    generator.manual_seed(SEED)

    points = torch.randn((NUM_POINTS, NUM_COORDS), generator=generator)
    features = points.clone()

    pointcloud_sample = BatchedData(x=features, pos=points)
    samples = [pointcloud_sample, pointcloud_sample.clone()]

    loader = tg_data.DataLoader(samples, batch_size=len(samples), shuffle=False, drop_last=False)

    batch = next(iter(loader))

    return batch
