import torch


class FeaturesFromPos:
    def __call__(self, data):
        data.x = data.pos.clone()
        return data


class FusePosAndNormals:
    def __call__(self, data):
        data.x = torch.cat((data.x, data.normal), dim=-1)
        del data.normal
        return data
