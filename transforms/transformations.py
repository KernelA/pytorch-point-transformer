import torch

from torch_geometric.transforms import SamplePoints


class FeaturesFromPos:
    def __call__(self, data):
        data.x = data.pos.clone()
        return data


class FusePosAndNormals:
    def __call__(self, data):
        data.x = torch.cat((data.x, data.normal), dim=-1)
        del data.normal
        return data


class TestPointSample:
    def __init__(self, num_points: int = 1024):
        self.point_sampler = SamplePoints(num=num_points)

    def __call__(self, data) -> bool:
        self.point_sampler(data)
        return True
