import zipfile
import pathlib
import pickle
from typing import Dict
import logging

import torch
from torch_geometric import loader, data
from torch_geometric.data import Data
from tqdm import tqdm
import trimesh
from trimesh.exchange import obj
from pytorch_lightning import LightningDataModule

from .dataloader_settings import LoadSettings


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
        self._logger = logging.getLogger()
        super().__init__(root=data_root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def download(self):
        pass

    def class_mapping(self) -> Dict[str, int]:
        return {class_name: i for i, class_name in enumerate(self.CLASS_NAMES)}

    def process(self):
        data_list = []

        class_mapping = self.class_mapping()

        with zipfile.ZipFile(self.path_to_zip, "r") as zip_archive:
            zip_items = zip_archive.infolist()
            for zip_info in tqdm(zip_items, desc="Process data"):
                if zip_info.is_dir() or zip_info.filename.startswith("__MACOSX"):
                    continue

                full_path = pathlib.Path(zip_info.filename)

                split_type = full_path.parent.name
                class_name = full_path.parent.parent.name

                if split_type == self.split_type:
                    with zip_archive.open(zip_info) as file:
                        loaded_data = trimesh.Trimesh(**obj.load_obj(file, "obj"))

                    mesh_data = Data(
                        pos=torch.from_numpy(loaded_data.vertices),
                        face=torch.from_numpy(loaded_data.faces.T),
                        y=class_mapping[class_name])

                    data_list.append(mesh_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0], pickle_protocol=pickle.HIGHEST_PROTOCOL)


class SimpleShapesDataset(LightningDataModule):
    def __init__(self,
                 *,
                 data_root: str,
                 path_to_zip: str,
                 train_load_sett: LoadSettings,
                 test_load_sett: LoadSettings):
        super().__init__()
        self.data_root = data_root
        self.path_to_zip = path_to_zip
        self.train_load_set = train_load_sett
        self.test_load_sett = test_load_sett
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        self.train_dataset = SimpleShapesInMemory(
            data_root=self.data_root,
            path_to_zip=self.path_to_zip,
            split_type="train",
            transform=self.train_load_set.transform,
            pre_filter=self.train_load_set.pre_filter,
            pre_transform=self.train_load_set.pre_transform)

        self.test_dataset = SimpleShapesInMemory(
            path_to_zip=self.path_to_zip,
            data_root=self.data_root,
            split_type="test",
            transform=self.test_load_sett.transform,
            pre_filter=self.test_load_sett.pre_filter,
            pre_transform=self.test_load_sett.pre_transform)

    def get_class_mapping(self) -> Dict[str, int]:
        return self.train_dataset.class_mapping()

    def train_dataloader(self):
        return loader.DataLoader(self.train_dataset,
                                 batch_size=self.train_load_set.batch_size,
                                 shuffle=True, drop_last=True, pin_memory=True,
                                 num_workers=self.train_load_set.num_workers)

    def val_dataloader(self):
        return loader.DataLoader(self.test_dataset,
                                 batch_size=self.test_load_sett.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True,
                                 num_workers=self.test_load_sett.num_workers)
