import pathlib

import hydra
from torch_geometric import transforms
from transforms import TestPointSample


def get_pre_transform():
    return transforms.Compose([
        transforms.NormalizeScale()
    ])


def get_pre_filter():
    return transforms.Compose([
        TestPointSample(num_points=1024)
    ])


@ hydra.main()
def main(config):
    out_dir = pathlib.Path(config.data_root)
    out_dir.mkdir(exist_ok=True, parents=True)

    pre_transform = get_pre_transform()
    pre_filter = get_pre_filter()

    for split_type in ("train", "test"):
        dataset = hydra.utils.instantiate(
            config, split_type=split_type,  pre_transform=pre_transform, pre_filter=pre_filter)


if __name__ == "__main__":
    main()
