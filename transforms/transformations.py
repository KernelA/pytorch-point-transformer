import torch


class FeaturesFromPos:
    def __call__(self, data):
        data.x = data.pos.clone()
        return data


class FusePosNormals:
    def __call__(self, data):
        data.x = torch.hstack((data.x, data.normal))
        del data.normal
        return data
