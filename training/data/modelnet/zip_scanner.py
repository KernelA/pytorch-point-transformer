import pathlib
import zipfile
from typing import Generator
import os
from collections import namedtuple


ItemInfo = namedtuple("ItemInfo", ["zipinfo", "split_type", "class_name"])


class ModelNetZipScanner:
    def __init__(self, zip_file: zipfile.ZipFile):
        self._zip_file = zip_file

    def num_objects(self) -> int:
        total_records = 0

        for _ in self.get_zip_info():
            total_records += 1

        return total_records

    def get_zip_info(self) -> Generator[ItemInfo, None, None]:
        zip_files = self._zip_file.infolist()

        for zip_file in zip_files:
            if zip_file.is_dir() or zip_file.filename.startswith("__MACOSX"):
                continue

            full_path = pathlib.Path(zip_file.filename)
            split_type = full_path.parent.name
            class_name = full_path.parent.parent.name

            if os.path.splitext(zip_file.filename)[1] == ".off":
                yield ItemInfo(zip_file, split_type, class_name)
