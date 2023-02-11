from typing import Dict, List
import os
import pathlib
import logging
import shutil

import fs
from fs.base import FS
import torch
from tqdm.auto import tqdm
from torch_geometric import loader
from torch_geometric import data
import trimesh
from joblib import Parallel, delayed
from pytorch_lightning import LightningDataModule

from .modelnet_info import MODELNET_CLASSES, ModelNetType
from ..dataloader_settings import LoadSettings


def _process_data(path: str, out_path: str, pre_transform, pre_filter, class_mapping: dict, index: int):
    loaded_data = trimesh.load(path)
    class_name = pathlib.Path(path).parent.parent.name

    new_mesh_data = data.Data(
        pos=torch.from_numpy(loaded_data.vertices),
        face=torch.from_numpy(loaded_data.faces.T),
        y=class_mapping[class_name])

    try:
        if pre_transform is not None:
            new_mesh_data = pre_transform(new_mesh_data)

        if pre_filter is not None:
            if not pre_filter(new_mesh_data):
                raise RuntimeError("Filter condition")

        torch.save(new_mesh_data, out_path)
    except RuntimeError:
        return index

    return None


def remove_refix(string: str, prefix: str):
    """For Colab Python 3.7
    """
    if string.startswith(prefix):
        return string[len(prefix):]

    return string


class ModelNet(data.Dataset):
    SPLIT_TYPES = ("train", "test")

    def __init__(self,
                 file_loc: str,
                 protocol: str,
                 data_root: str,
                 split_type: str,
                 class_names: List[str],
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 n_jobs: int = -1):
        assert split_type in self.SPLIT_TYPES
        self.logger = logging.getLogger()
        self.split_type = split_type
        self.class_names = class_names

        file_url = f"{protocol}{file_loc}"

        with fs.open_fs(file_url, writeable=False) as filesystem:
            self._raw_fs_paths = self._get_mesh_names(filesystem)

        self._file_url = file_url
        self.n_jobs = n_jobs
        self._processed_file_names = [os.path.join(
            self.split_type, str(pathlib.Path(remove_refix(abs_path, "/")).with_suffix(".pt"))) for abs_path in self._raw_fs_paths]

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

    def _get_mesh_names(self, filesystem: FS) -> List[str]:
        filtered_files = []

        for file_path in filesystem.walk.files(exclude_dirs=["__MACOSX*"]):
            if file_path.startswith("__MACOSX"):
                continue

            full_path = pathlib.Path(file_path)

            if full_path.parent.name == self.split_type and full_path.suffix == ".off":
                filtered_files.append(file_path)

        return filtered_files

    def process(self):
        self.logger.info("Open %s.", self._file_url)

        temp_paths = []

        with fs.open_fs(self._file_url, writeable=False) as filesystem:
            for file_path in tqdm(self._raw_fs_paths, desc="Copy files"):
                out_file = os.path.join(self.raw_dir, file_path.removeprefix("/"))

                os.makedirs(os.path.dirname(out_file), exist_ok=True)

                with open(out_file, "wb") as tmp_fie:
                    filesystem.download(file_path, tmp_fie)
                    temp_paths.append(out_file)

        try:
            for path in self.processed_paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            delete_indices: list = Parallel(n_jobs=self.n_jobs, prefer="processes", verbose=1)(
                delayed(_process_data)(
                    src_path,
                    out_path,
                    self.pre_transform,
                    self.pre_filter,
                    self.get_class_mapping(),
                    i
                ) for i, (src_path, out_path) in enumerate(
                    zip(temp_paths, self.processed_paths)
                )
            )
        finally:
            shutil.rmtree(self.raw_dir)

        delete_indices = set(filter(lambda x: x is not None, delete_indices))
        num_unprocessed = len(delete_indices)
        self._processed_file_names = [path for i, path in enumerate(
            self._processed_file_names) if i not in delete_indices]

        if num_unprocessed > 0:
            self.logger.exception(
                "Unexpected error in pre_transform or transforms. Remove %d files from data", num_unprocessed)
        self.logger.info("Processed %d", len(self._processed_file_names))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx: int):
        return torch.load(self.processed_paths[idx])


class ModelNet40(ModelNet):
    def __init__(self,
                 file_loc: str,
                 protocol: str,
                 data_root: str,
                 split_type: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 n_jobs: int = -1):
        super().__init__(file_loc,
                         protocol,
                         data_root,
                         split_type,
                         MODELNET_CLASSES[ModelNetType.modelnet_40],
                         transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter,
                         n_jobs=n_jobs)

    def process(self):
        return super().process()


class ModelNet10(ModelNet):
    def __init__(self,
                 file_loc: str,
                 protocol: str,
                 data_root: str,
                 split_type: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 n_jobs: int = -1):
        super().__init__(file_loc, protocol, data_root, split_type, MODELNET_CLASSES[ModelNetType.modelnet_10],
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter, n_jobs=n_jobs)

    def process(self):
        return super().process()


class ModelNetDataset(LightningDataModule):
    def __init__(self,
                 *,
                 data_root: str,
                 protocol: str,
                 file_loc: str,
                 dataset_type: ModelNetType,
                 train_load_sett: LoadSettings,
                 test_load_sett: LoadSettings,
                 n_jobs: int = -1):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_type = dataset_type
        self.train_load_sett = train_load_sett
        self.test_load_sett = test_load_sett
        self._file_loc = file_loc
        self._protocol = protocol
        self._data_root = data_root
        self._n_jobs = n_jobs

    def setup(self, stage):
        cls = ModelNet10 if self.dataset_type == ModelNetType.modelnet_10 else ModelNet40

        if stage == "fit" or stage is None:
            self.train_dataset = cls(file_loc=self._file_loc, protocol=self._protocol, data_root=self._data_root,
                                     split_type="train",
                                     n_jobs=self._n_jobs,
                                     transform=self.train_load_sett.transform,
                                     pre_transform=self.train_load_sett.pre_transform,
                                     pre_filter=self.train_load_sett.pre_filter)

        if stage == "validate" or stage is None:
            self.val_dataset = cls(file_loc=self._file_loc, protocol=self._protocol, data_root=self._data_root,
                                   split_type="test",
                                   n_jobs=self._n_jobs,
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
