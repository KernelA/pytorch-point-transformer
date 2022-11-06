import zipfile
from collections import namedtuple

import numpy as np
from torch_geometric.io.off import parse_off

MeshData = namedtuple("MeshData", ["vertices", "faces"])


class OffConverter:
    def convert(self, zip_archive: zipfile.ZipFile, zipinfo: zipfile.ZipInfo) -> MeshData:
        mesh_data = parse_off(zip_archive.read(zipinfo).decode("utf-8").splitlines()[:-1])

        return MeshData(mesh_data.pos.numpy().astype(np.float32), mesh_data.face.T.numpy())
