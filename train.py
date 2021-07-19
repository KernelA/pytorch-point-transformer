import hydra
from hydra.core.config_store import ConfigStore
from torch_geometric import transforms
import data

from data_configs import TrainConfig, DataConfig
from data import SimpleShapesDataset

cs = ConfigStore().instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="data", node=DataConfig)


def get_train_transform(num_points: int):
    return transforms.Compose([
        transforms.SamplePoints(num=num_points),
        transforms.NormalizeScale()]
    )


@ hydra.main(config_name="train")
def main(config: TrainConfig):
    pre_transform = get_train_transform(config.data.num_points)

    SimpleShapesDataset(config.data.data_root, config.train_batch_size, config.test_batch_size,
                        train_pre_transform=pre_transform, test_pre_transform=pre_transform)


if __name__ == "__main__":
    main()
