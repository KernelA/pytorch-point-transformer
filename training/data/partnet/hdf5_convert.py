from typing import List

import h5py
import hdf5plugin
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm


class HdfIO:

    @staticmethod
    def save_dataset(hdf_file, key, numpy_data):
        chunk_size = 1

        compression_opts = hdf5plugin.SZ3(absolute=1e-6)

        if isinstance(numpy_data.dtype, np.integer):
            compression_opts = hdf5plugin.Zfp(reversible=True)

        max_shape = (None,) + numpy_data.shape[1:]

        hdf_file.create_dataset(
            key,
            shape=numpy_data.shape,
            dtype=numpy_data.dtype,
            data=numpy_data,
            chunks=(chunk_size,) + numpy_data.shape[1:],
            maxshape=max_shape,
            **compression_opts
        )

    @staticmethod
    def union_and_compress(path_to_files: List[str], path_to_new_file: str, transforms):
        total_items = 0

        with h5py.File(path_to_new_file, "w") as new_file:
            temp_data_storage = {}

            for file_path in tqdm(path_to_files):
                with h5py.File(file_path, "r") as origin_file:

                    data_samples = []

                    for key, orig_dataset in origin_file.items():
                        if not data_samples:
                            data_samples = [Data() for _ in range(orig_dataset.shape[0])]

                        if key == "label_seg":
                            numpy_data = orig_dataset[:].astype(np.uint8)
                        elif key == "data_num":
                            numpy_data = orig_dataset[:].astype(np.int16)
                        else:
                            numpy_data = orig_dataset[:]

                        for i in range(numpy_data.shape[0]):
                            new_key = key

                            if new_key == "data":
                                new_key = "pos"

                            if new_key == "data_num":
                                data = torch.from_numpy(numpy_data[i].reshape(1))
                            else:
                                data = torch.from_numpy(numpy_data[i])

                            setattr(data_samples[i], new_key, data)

                    batch = Batch.from_data_list(data_samples)
                    total_items += batch.num_graphs

                    if transforms is not None:
                        batch = transforms(batch)

                    data_samples = batch.to_data_list()

                    temp_data_storage = {
                        key: np.concatenate([getattr(data, key).numpy()[None, ...]
                                            for data in data_samples], axis=0)
                        for key in data_samples[0].keys
                    }

                    for key, numpy_data in temp_data_storage.items():
                        if key in new_file:
                            old_dataset = new_file[key]
                            old_length = old_dataset.shape[0]
                            old_dataset.resize(old_length + numpy_data.shape[0], axis=0)
                            old_dataset[old_length:] = numpy_data
                        else:
                            HdfIO.save_dataset(new_file, key, numpy_data)

            new_file.flush()

            for key, dataset in new_file.items():
                real_samples = dataset.shape[0]

                if real_samples != total_items:
                    raise ValueError(
                        f"Incorrect conversion detected. Total number of samples is less than in the files: {real_samples} != {total_items}")
