import torch
import pytest
from torch_geometric import data

NUM_COORDS = 3
NUM_POINTS = 10
SEED = 20


@pytest.fixture()
def sample_batch():
    generator = torch.Generator()
    generator.manual_seed(SEED)

    points = torch.randn((NUM_POINTS, NUM_COORDS), generator=generator)
    features = points.clone()

    pointcloud_sample = data.Data(x=features, pos=points)
    samples = [pointcloud_sample, pointcloud_sample.clone()]

    batch = data.Batch.from_data_list(samples)
    return batch
