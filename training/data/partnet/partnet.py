import logging
import os
from typing import Dict

import h5py
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data, DataLoader, Dataset
from tqdm.auto import tqdm

from ..dataloader_settings import LoadSettings


class PartNet(Dataset):
    """https://arxiv.org/pdf/1812.02713.pdf (5.1. Fine-grained Semantic Segmentation)
    """

    def __init__(self,
                 dataset_dir: str,
                 data_root: str,
                 split_type: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        if pre_filter is not None:
            raise ValueError("pre_filter is not supported")

        self.logger = logging.getLogger()

        assert split_type in ("train", "val"), "Incorrect split type. Expected: 'train', 'val'"

        self._file_loc = os.path.join(dataset_dir, f"{split_type}.h5")

        if not os.path.isfile(self._file_loc):
            raise FileNotFoundError(f"Cannot find: '{self._file_loc}'")

        step = 2048
        self._unique_labels = set()

        with h5py.File(self._file_loc, "r") as f:
            for i in range(0, f["label_seg"].shape[0], step):
                self._unique_labels.update(map(int, f["label_seg"][i: i + step].flat))

            self.logger.info("Found %d labels in the %s", len(self._unique_labels), split_type)
            self._num_items = f["label_seg"].shape[0]

        self._split_type = split_type
        self._processed_file_names = [os.path.join(
            self._split_type, f"sample_{i}.pt") for i in range(self._num_items)]
        super().__init__(data_root, transform, pre_transform, pre_filter)

    def close(self):
        pass

    def get_class_mapping(self) -> Dict[str, int]:
        return {i: i for i in self._unique_labels}

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def download(self):
        self.logger.warning("Cannot automatically download dataset because of license restriction")

    def process(self):
        with h5py.File(self._file_loc, "r") as f:
            for i, full_path in tqdm(enumerate(self.processed_paths), total=len(self.processed_paths)):
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                positions = torch.from_numpy(f["pos"][i])
                point_labels = torch.from_numpy(f["label_seg"][i])

                data = Data(pos=positions, y=point_labels)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, full_path)

    def __len__(self):
        return self._num_items

    def __getitem__(self, idx):
        data = torch.load(self.processed_paths[idx])

        if self.transform is not None:
            data = self.transform(data)

        return data


class PartNetDataset(LightningDataModule):
    def __init__(self,
                 *,
                 data_root: str,
                 dataset_dir: str,
                 train_load_sett: LoadSettings,
                 test_load_sett: LoadSettings):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.train_load_sett = train_load_sett
        self.test_load_sett = test_load_sett
        self._dataset_dir = dataset_dir
        self._data_root = data_root

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = PartNet(dataset_dir=self._dataset_dir, data_root=self._data_root,
                                         transform=self.train_load_sett.transform,
                                         split_type="train",
                                         pre_transform=self.train_load_sett.pre_transform,
                                         pre_filter=self.train_load_sett.pre_filter)

        if stage == "validate" or stage is None:
            self.val_dataset = PartNet(dataset_dir=self._dataset_dir,
                                       data_root=self._data_root,
                                       split_type="val",
                                       transform=self.train_load_sett.transform,
                                       pre_transform=self.train_load_sett.pre_transform,
                                       pre_filter=self.train_load_sett.pre_filter)

    def get_class_mapping(self):
        if self.train_dataset is not None:
            return self.train_dataset.get_class_mapping()
        if self.val_dataset is not None:
            return self.val_dataset.get_class_mapping()
        raise RuntimeError("You need setup dataset first")

    def teardown(self, stage: str) -> None:
        if stage == "fit" and self.train_dataset is not None:
            self.train_dataset.close()
        elif stage == "validate" and self.val_dataset is not None:
            self.val_dataset.close()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          pin_memory=True,
                          batch_size=self.train_load_sett.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.train_load_sett.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.test_load_sett.batch_size,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True,
                          num_workers=self.test_load_sett.num_workers)
