import pathlib

import hydra
from torch_geometric import transforms


def get_pre_transform():
    return transforms.Compose([
        transforms.NormalizeScale()
    ])


@hydra.main()
def main(config):
    out_dir = pathlib.Path(config.data_root)
    out_dir.mkdir(exist_ok=True, parents=True)

    pre_transform = get_pre_transform()

    for split_type in ("train", "test"):
        dataset = hydra.utils.instantiate(
            config, split_type=split_type,  pre_transform=pre_transform)


if __name__ == "__main__":
    main()
