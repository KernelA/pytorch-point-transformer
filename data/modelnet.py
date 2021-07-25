from typing import Dict, List
import os
import shutil
import zipfile
import pathlib
import logging

import torch
from torch_geometric import data
from torch_geometric.io.off import parse_off
import tqdm

import log_set
from .mesh_data import BatchedData


class ModelNet(data.InMemoryDataset):
    SPLIT_TYPES = ("train", "test")

    def __init__(self,
                 path_to_zip: str,
                 data_root: str,
                 split_type: str,
                 class_names: List[str],
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert split_type in self.SPLIT_TYPES
        self.logger = logging.getLogger("pt.data")
        self.path_to_zip = path_to_zip
        self.split_type = split_type
        self.class_names = class_names
        self._processed_file_names = [f"{split_type}.pt"]
        super().__init__(root=data_root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.path_to_zip)]

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def download(self):
        self.logger.info("Copy %s to %s", self.path_to_zip, self.raw_dir)
        shutil.copy(self.path_to_zip, self.raw_dir)

    def get_class_mapping(self) -> Dict[str, int]:
        return {class_name: i for i, class_name in enumerate(self.class_names)}

    def process(self):
        data_list = []

        class_mapping = self.get_class_mapping()

        zip_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        self.logger.info("Open %s", zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_archive:
            zip_files = zip_archive.infolist()

            for zip_file in tqdm.tqdm(zip_files, total=len(zip_files), desc="Process files"):
                if zip_file.is_dir() or zip_file.filename.startswith("__MACOSX"):
                    continue

                full_path = pathlib.Path(zip_file.filename)

                if full_path.parent.name == self.split_type and full_path.suffix == ".off":
                    try:
                        mesh_data = parse_off(zip_archive.read(
                            zip_file).decode("utf-8").splitlines()[:-1])
                    except UnicodeDecodeError:
                        self.logger.exception("Error when parse %s. Skip it", zip_file.filename)
                        continue

                    new_mesh_data = BatchedData()
                    for key in mesh_data.keys:
                        setattr(new_mesh_data, key, mesh_data[key])
                    del mesh_data
                    new_mesh_data.y = class_mapping[full_path.parent.parent.name]
                    new_mesh_data.name = full_path.name
                    data_list.append(new_mesh_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        transformed = []

        if self.pre_transform is not None:
            for i in tqdm.trange(len(data_list)):
                try:
                    transformed.append(self.pre_transform(data_list[i]))
                except RuntimeError:
                    self.logger.exception("Unexpected error in pre_transform")

        self.logger.info("Loaded %d samples", len(transformed))

        data, slices = self.collate(transformed)
        torch.save((data, slices), self.processed_paths[0])


class ModelNet40(ModelNet):
    def __init__(self,
                 path_to_zip: str,
                 data_root: str,
                 split_type: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        class_names = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
                       "chair", "cone", "cup", "curtain",
                       "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard",
                       "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant",
                       "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet",
                       "tv_stand", "vase", "wardrobe", "xbox"]
        super().__init__(path_to_zip, data_root, split_type, class_names,
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def download(self):
        return super().download()

    def process(self):
        return super().process()


class ModelNet10(ModelNet):
    def __init__(self,
                 path_to_zip: str,
                 data_root: str,
                 split_type: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        class_names = ["bathtub", "bed", "chair", "desk", "dresser",
                       "monitor", "night_stand", "sofa", "table", "toilet"]
        super().__init__(path_to_zip, data_root, split_type, class_names,
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def download(self):
        return super().download()

    def process(self):
        return super().process()


class ModelNetDataset:
    def __init__(self,
                 *,
                 data_root: str,
                 path_to_zip: str,
                 name: str,
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
        assert name in ("40", "10")

        class_name = ModelNet40 if name == "40" else ModelNet10

        self.train_dataset = class_name(
            data_root=data_root, path_to_zip=path_to_zip,
            split_type="train",
            transform=train_transform,
            pre_filter=train_pre_filter,
            pre_transform=train_pre_transform)

        self.test_dataset = class_name(
            data_root=data_root, path_to_zip=path_to_zip,
            split_type="test",
            pre_transform=test_pre_transform,
            transform=test_transform,
            pre_filter=test_pre_filter)

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


class ModelNet40Dataset(ModelNetDataset):
    def __init__(self, *, data_root: str,
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
        super().__init__(data_root=data_root,
                         name="40",
                         path_to_zip=path_to_zip,
                         train_num_workers=train_num_workers,
                         test_num_workers=test_num_workers,
                         train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size,
                         train_transform=train_transform,
                         test_transform=test_transform,
                         train_pre_filter=train_pre_filter,
                         test_pre_filter=test_pre_filter,
                         train_pre_transform=train_pre_transform,
                         test_pre_transform=test_pre_transform)


class ModelNet10Dataset(ModelNetDataset):
    def __init__(self, *, data_root: str,
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
        super().__init__(data_root=data_root,
                         name="10",
                         path_to_zip=path_to_zip,
                         train_num_workers=train_num_workers,
                         test_num_workers=test_num_workers,
                         train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size,
                         train_transform=train_transform,
                         test_transform=test_transform,
                         train_pre_filter=train_pre_filter,
                         test_pre_filter=test_pre_filter,
                         train_pre_transform=train_pre_transform,
                         test_pre_transform=test_pre_transform)
