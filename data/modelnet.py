from typing import Dict
import os
import shutil
import zipfile
import pathlib
import traceback

import torch
from torch_geometric import data
from torch_geometric.io.off import parse_off
import tqdm


class ModelNet40(data.InMemoryDataset):
    CLASS_NAMES = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
                   "chair", "cone", "cup", "curtain",
                   "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard",
                   "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant",
                   "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet",
                   "tv_stand", "vase", "wardrobe", "xbox"]
    SPLIT_TYPES = ("train", "test")

    def __init__(self,
                 path_to_zip: str,
                 data_root: str,
                 split_type: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert split_type in self.SPLIT_TYPES
        assert len(self.CLASS_NAMES) == 40
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

    def get_class_mapping(self) -> Dict[str, int]:
        return {class_name: i for i, class_name in enumerate(self.CLASS_NAMES)}

    def process(self):
        data_list = []

        class_mapping = self.get_class_mapping()

        with zipfile.ZipFile(os.path.join(self.raw_dir, self.raw_file_names[0]), "r") as zip_archive:
            files = zip_archive.infolist()
            for file in tqdm.tqdm(files, total=len(files), desc="Process files"):
                if not file.is_dir():
                    full_path = pathlib.Path(file.filename)
                    if full_path.parent.name == self.split_type and full_path.suffix == ".off":
                        mesh_data = parse_off(zip_archive.read(
                            file).decode("ascii").splitlines()[:-1])
                        mesh_data.y = class_mapping[full_path.parent.parent.name]
                        mesh_data.name = full_path.name
                        data_list.append(mesh_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        transformed = []

        if self.pre_transform is not None:
            for i in tqdm.trange(len(data_list)):
                try:
                    transformed.append(self.pre_transform(data_list[i]))
                except RuntimeError:
                    print(traceback.format_exc())

        data, slices = self.collate(transformed)
        torch.save((data, slices), self.processed_paths[0])


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
                 train_pre_transform,
                 test_pre_transform):
        assert name == "40", "ModelNet 10 is not implemented"

        self.train_dataset = ModelNet40(
            data_root=data_root, path_to_zip=path_to_zip, split_type="train", pre_transform=train_pre_transform)

        self.test_dataset = ModelNet40(
            data_root=data_root, path_to_zip=path_to_zip, split_type="test", pre_transform=test_pre_transform)

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
                 train_pre_transform,
                 test_pre_transform):
        super().__init__(data_root=data_root,
                         name="40",
                         path_to_zip=path_to_zip,
                         train_num_workers=train_num_workers,
                         test_num_workers=test_num_workers,
                         train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size,
                         train_pre_transform=train_pre_transform,
                         test_pre_transform=test_pre_transform)
