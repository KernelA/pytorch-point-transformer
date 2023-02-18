import torch
import pytest

from torch_geometric.data import Batch, Data

NUM_COORDS = 3
NUM_POINTS = 10
SEED = 20


@pytest.fixture(scope="function")
def generator():
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return generator


@pytest.fixture(scope="session")
def session_generator():
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return generator


@pytest.fixture(scope="session")
def sample_batch(session_generator):
    points = torch.randn((NUM_POINTS, NUM_COORDS), generator=session_generator)
    features = points.clone()

    pointcloud_sample = Data(x=features, pos=points)

    samples = [pointcloud_sample, pointcloud_sample.clone()]

    return Batch.from_data_list(samples)
