import os
import shutil
import zipfile
import pathlib
from typing import Dict

import torch
from torch_geometric import data

from .mesh_data import BatchedData
from .io_utils import read_obj_from_bytes


class SimpleShapesInMemory(data.InMemoryDataset):
    CLASS_NAMES = ["cone", "cube", "cylinder", "plane", "torus", "uv_sphere"]
    SPLIT_TYPES = ("train", "test", "valid")

    def __init__(self,
                 path_to_zip: str,
                 data_root: str,
                 split_type: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert split_type in self.SPLIT_TYPES
        self.path_to_zip = path_to_zip
        self.split_type = split_type
        self._processed_file_names = [f"{split_type}.pt"]
        super().__init__(root=data_root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.path_to_zip)]

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def download(self):
        shutil.copy(self.path_to_zip, self.raw_dir)

    def class_mapping(self) -> Dict[str, int]:
        return {class_name: i for i, class_name in enumerate(self.CLASS_NAMES)}

    def process(self):
        data_list = []

        class_mapping = self.class_mapping()

        with zipfile.ZipFile(os.path.join(self.raw_dir, self.raw_file_names[0]), "r") as zip_archive:
            files = zip_archive.infolist()
            for file in files:
                if not file.is_dir():
                    full_path = pathlib.Path(file.filename)
                    if full_path.parent.name == self.split_type:
                        mesh_data = read_obj_from_bytes(zip_archive.read(file).decode("ascii"))
                        new_mesh_data = BatchedData()
                        for key in mesh_data.keys:
                            setattr(new_mesh_data, key, mesh_data[key])

                        del mesh_data
                        new_mesh_data.y = class_mapping[full_path.parent.parent.name]
                        data_list.append(new_mesh_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SimpleShapesDataset:
    def __init__(self,
                 *,
                 data_root: str,
                 path_to_zip: str,
                 train_num_workers: int,
                 test_num_workers: int,
                 train_batch_size: int,
                 test_batch_size: int,
                 train_transform=None,
                 test_transform=None,
                 train_pre_filter=None,
                 test_pre_filter=None,
                 train_pre_transform=None,
                 test_pre_transform=None):

        self.train_dataset = SimpleShapesInMemory(
            data_root=data_root,
            path_to_zip=path_to_zip,
            split_type="train",
            transform=train_transform,
            pre_filter=train_pre_filter,
            pre_transform=train_pre_transform)

        self.test_dataset = SimpleShapesInMemory(
            path_to_zip=path_to_zip,
            data_root=data_root,
            split_type="test",
            transform=test_transform,
            pre_filter=test_pre_filter,
            pre_transform=test_pre_transform)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers

    def get_class_mapping(self) -> Dict[str, int]:
        return self.train_dataset.class_mapping()

    def get_train_loader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                               shuffle=True, drop_last=True, pin_memory=True,
                               num_workers=self.train_num_workers)

    def get_test_loader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                               shuffle=False, drop_last=False, pin_memory=True,
                               num_workers=self.test_num_workers)
