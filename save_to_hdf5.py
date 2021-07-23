import zipfile
import pathlib

import numpy as np
import h5py
import hydra
import trimesh
from trimesh import resolvers
import tqdm
from hydra.core.config_store import ConfigStore

from data_configs import ConvertConfig

cs = ConfigStore().instance()
cs.store("convert", ConvertConfig)


@hydra.main(config_name="convert")
def main(config: ConvertConfig):
    hdf5_path = pathlib.Path(config.path_to_hdf5)
    hdf5_path.parent.mkdir(exist_ok=True, parents=True)

    with h5py.File(hdf5_path, "w") as hdf_file, zipfile.ZipFile(config.path_to_archive, "r") as zip_archive:
        zip_files = tuple(filter(lambda x: not x.is_dir(), zip_archive.infolist()))

        for zip_file in tqdm.tqdm(zip_files):
            full_path = pathlib.Path(zip_file.filename)
            if full_path.suffix in config.extensions:
                with zip_archive.open(zip_file, "r") as mesh_file:
                    mesh = trimesh.load(mesh_file, file_type=config.mesh_format,
                                        resolver=resolvers.FilePathResolver(""))

                if mesh.is_empty:
                    print("Skip because it is empty: ", zip_file.filename)
                    continue

                dataset_name = "/".join(("/", *full_path.parts[-3:-1], full_path.stem))
                hdf_file.create_dataset(
                    f"{dataset_name}/vertices", data=np.array(mesh.vertices), chunks=(1, mesh.vertices.shape[1]))
                hdf_file.create_dataset(f"{dataset_name}/faces",
                                        data=np.array(mesh.faces), chunks=(1, mesh.faces.shape[1]))


if __name__ == "__main__":
    main()
