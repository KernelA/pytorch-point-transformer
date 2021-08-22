from typing import Dict, List
import os
import zipfile
import pathlib
import logging

import torch
from torch_geometric import data
from torch_geometric.io.off import parse_off
import tqdm

import log_set
from .mesh_data import BatchedData


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
        self.logger = logging.getLogger("pt.data")
        self.split_type = split_type
        self.class_names = class_names

        self._path_to_zip = path_to_zip

        with zipfile.ZipFile(self._path_to_zip, "r") as archive:
            zip_names = self._get_zip_names(archive)
            self._processed_file_names = list(
                f"mesh_{self.split_type}_{i}.pt" for i in range(len(zip_names)))

        self._raw_file_name = [os.path.basename(self._path_to_zip)]

        super().__init__(root=data_root,
                         transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return self._raw_file_name

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
            zip_files = self._get_zip_names(zip_archive)

            for zip_file, processed_file in tqdm.tqdm(zip(zip_files, self.processed_paths), total=len(zip_files), desc="Process files"):
                full_path = pathlib.Path(zip_file.filename)

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

                try:
                    if self.pre_transform is not None:
                        transformed = self.pre_transform(new_mesh_data)
                    else:
                        transformed = new_mesh_data

                    if self.pre_filter is not None:
                        if not self.pre_filter(transformed):
                            raise RuntimeError("Filter condition")

                    torch.save(transformed, processed_file)
                    num_samples += 1
                except RuntimeError:
                    self._processed_file_names.remove(os.path.basename(processed_file))
                    self.logger.exception(
                        "Unexpected error in pre_transform or transforms. Remove %s from data", zip_file)

        self.logger.info("Processed %d", num_samples)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx: int) -> BatchedData:
        return torch.load(self.processed_paths[idx])


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
                 test_pre_transform=None,):
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
            pre_filter=test_pre_filter,
            pre_transform=test_pre_transform,
            transform=test_transform)

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
