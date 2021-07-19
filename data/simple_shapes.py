import os

from torch_geometric import datasets, data


class SimpleShapesDataset:
    def __init__(self,
                 data_root: str,
                 train_batch: int,
                 test_batch: int,
                 train_pre_transform,
                 test_pre_transform):
        self.train_dataset = datasets.GeometricShapes(
            root=data_root, pre_transform=train_pre_transform)
        self.test_dataset = datasets.GeometricShapes(
            data_root, train=False, pre_transform=test_pre_transform)
        self.train_batch = train_batch
        self.test_batch = test_batch

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.train_batch, shuffle=True, pin_memory=True, drop_last=True)

    def test_loader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.test_batch, shuffle=False, pin_memory=True, drop_last=False)
