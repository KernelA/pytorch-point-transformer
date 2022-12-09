from typing import Dict, List
import os
import zipfile
import pathlib
import logging

import torch
from torch_geometric import loader
from tqdm.auto import tqdm
import trimesh
from trimesh.exchange.off import load_off
from pytorch_lightning import LightningDataModule

from .modelnet_info import MODELNET_CLASSES, ModelNetType
from ..dataloader_settings import LoadSettings


import log_set


class ModelNet(data.Dataset):
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
        self.logger = logging.getLogger()
        self.split_type = split_type
        self.class_names = class_names

        self._path_to_zip = path_to_zip

        with zipfile.ZipFile(self._path_to_zip, "r") as archive:
            zip_names = self._get_zip_names(archive)
            self._processed_file_names = list(
                f"mesh_{self.split_type}_{i}.pt" for i in range(len(zip_names)))

        super().__init__(root=data_root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    def download(self):
        pass

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def get_class_mapping(self) -> Dict[str, int]:
        return {class_name: i for i, class_name in enumerate(self.class_names)}

    def _get_zip_names(self, zip_archive) -> List[zipfile.ZipInfo]:
        zip_files = zip_archive.infolist()
        filtered_zip_files = []

        for zip_file in zip_files:
            if zip_file.is_dir() or zip_file.filename.startswith("__MACOSX"):
                continue

            full_path = pathlib.Path(zip_file.filename)

            if full_path.parent.name == self.split_type and full_path.suffix == ".off":
                filtered_zip_files.append(zip_file)

        return filtered_zip_files

    def process(self):
        class_mapping = self.get_class_mapping()

        self.logger.info("Open %s", self._path_to_zip)

        num_samples = 0

        with zipfile.ZipFile(self._path_to_zip, "r") as zip_archive:
            zip_file_names = self._get_zip_names(zip_archive)

            for zip_file, processed_file in tqdm(
                    zip(zip_file_names, tuple(self.processed_paths)),
                    total=len(zip_file_names),
                    desc="Process files"):

                full_path = pathlib.Path(zip_file.filename)

                with zip_archive.open(zip_file, "r") as zip_file:
                    loaded_data = trimesh.Trimesh(**load_off(zip_file))

                class_name = full_path.parent.parent.name

                new_mesh_data = data.Data(pos=torch.from_numpy(loaded_data.vertices), face=torch.from_numpy(
                    loaded_data.faces.T), y=class_mapping[class_name])

                try:
                    if self.pre_transform is not None:
                        new_mesh_data = self.pre_transform(new_mesh_data)

                    if self.pre_filter is not None:
                        if not self.pre_filter(new_mesh_data):
                            raise RuntimeError("Filter condition")

                    torch.save(new_mesh_data, processed_file)
                    num_samples += 1
                except RuntimeError:
                    self._processed_file_names.remove(os.path.basename(processed_file))
                    self.logger.exception(
                        "Unexpected error in pre_transform or transforms. Remove %s from data", zip_file)

        self.logger.info("Processed %d", num_samples)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx: int):
        return torch.load(self.processed_paths[idx])


class ModelNet40(ModelNet):
    def __init__(self,
                 path_to_zip: str,
                 data_root: str,
                 split_type: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(path_to_zip,
                         data_root,
                         split_type,
                         MODELNET_CLASSES[ModelNetType.modelnet_40],
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

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
        super().__init__(path_to_zip, data_root, split_type, MODELNET_CLASSES[ModelNetType.modelnet_10],
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def process(self):
        return super().process()


class ModelNetDataset(LightningDataModule):
    def __init__(self,
                 *,
                 data_root: str,
                 path_to_zip: str,
                 dataset_type: ModelNetType,
                 train_load_sett: LoadSettings,
                 test_load_sett: LoadSettings):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_type = dataset_type
        self.train_load_sett = train_load_sett
        self.test_load_sett = test_load_sett
        self._path_to_zip = path_to_zip
        self._data_root = data_root

    def setup(self, stage):
        cls = ModelNet10 if self.dataset_type == ModelNetType.modelnet_10 else ModelNet40

        if stage == "fit" or stage is None:
            self.train_dataset = cls(path_to_zip=self._path_to_zip, data_root=self._data_root,
                                     split_type="train",
                                     transform=self.train_load_sett.transform,
                                     pre_transform=self.train_load_sett.pre_transform,
                                     pre_filter=self.train_load_sett.pre_filter)

        if stage == "validate" or stage is None:
            self.val_dataset = cls(path_to_zip=self._path_to_zip, data_root=self._data_root,
                                   split_type="test",
                                   transform=self.train_load_sett.transform,
                                   pre_transform=self.train_load_sett.pre_transform,
                                   pre_filter=self.train_load_sett.pre_filter)

    def get_class_mapping(self):
        if self.train_dataset is not None:
            return self.train_dataset.get_class_mapping()
        if self.val_dataset is not None:
            return self.val_dataset.get_class_mapping()
        raise RuntimeError("You need setup dataset first")

    def train_dataloader(self):
        return loader.DataLoader(self.train_dataset, batch_size=self.train_load_sett.batch_size,
                               shuffle=True, drop_last=True, pin_memory=True,
                               num_workers=self.train_load_sett.num_workers)

    def val_dataloader(self):
        return loader.DataLoader(self.val_dataset, batch_size=self.test_load_sett.batch_size,
                               shuffle=False, drop_last=False, pin_memory=True,
                               num_workers=self.test_load_sett.num_workers)
