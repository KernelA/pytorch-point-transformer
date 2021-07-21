from typing import Dict
import glob
import os

import torch
from torch_geometric.io import read_off
from torch_geometric import datasets, data


class ModelNetCategories(datasets.ModelNet):
    def __init__(self, root: str, name: str, train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self._class_mapping = dict()
        super().__init__(root=root, name=name,
                         train=train,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)

    def get_class_mapping(self) -> Dict[str, int]:
        return self._class_mapping

    def process_set(self, dataset):
        categories = glob.glob(os.path.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = os.path.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])
                data_list.append(data)

            self._class_mapping[category] = target

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)


class ModelNetDataset:
    def __init__(self,
                 *,
                 data_root: str,
                 name: str,
                 train_num_workers: int,
                 test_num_workers: int,
                 train_batch_size: int,
                 test_batch_size: int,
                 train_pre_transform,
                 test_pre_transform):

        self.train_dataset = datasets.ModelNet(
            data_root, name=name, train=True, pre_transform=train_pre_transform)

        self.test_dataset = datasets.ModelNet(
            data_root, name=name, train=False, pre_transform=test_pre_transform)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers

    def get_class_mapping(self):
        return self.train_dataset.get_class_mapping()

    def get_train_loader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                               shuffle=True, drop_last=True, pin_memory=True,
                               num_workers=self.train_num_workers)

    def get_test_loader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                               shuffle=False, drop_last=False, pin_memory=True,
                               num_workers=self.test_num_workers)


class ModelNet10Dataset(ModelNetDataset):
    def __init__(self, *, data_root: str,
                 train_num_workers: int,
                 test_num_workers: int,
                 train_batch_size: int,
                 test_batch_size: int,
                 train_pre_transform,
                 test_pre_transform):
        super().__init__(data_root=data_root,
                         name="10",
                         train_num_workers=train_num_workers,
                         test_num_workers=test_num_workers,
                         train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size,
                         train_pre_transform=train_pre_transform,
                         test_pre_transform=test_pre_transform)


class ModelNet40Dataset(ModelNetDataset):
    def __init__(self, *, data_root: str,
                 train_num_workers: int,
                 test_num_workers: int,
                 train_batch_size: int,
                 test_batch_size: int,
                 train_pre_transform,
                 test_pre_transform):
        super().__init__(data_root=data_root,
                         name="40",
                         train_num_workers=train_num_workers,
                         test_num_workers=test_num_workers,
                         train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size,
                         train_pre_transform=train_pre_transform,
                         test_pre_transform=test_pre_transform)
