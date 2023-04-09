import argparse
import pathlib
from collections import defaultdict

from torch_geometric.transforms import Compose, NormalizeScale
from tqdm.auto import tqdm

import log_set
from training.data.partnet import HdfIO
from transforms import FPS, PosToFloat32


def main(args):
    data_dir = pathlib.Path(args.data_dir)

    if not data_dir.is_dir():
        raise ValueError(f"'{data_dir}' is not a directory")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    dirs = list(data_dir.iterdir())

    progress = tqdm(dirs)

    merge_files = defaultdict(list)

    for sample_dir in progress:
        if not sample_dir.is_dir():
            continue

        progress.set_description(f"Scan: {sample_dir}")

        for file in sample_dir.iterdir():
            if file.stem not in ("test_files", "train_files", "val_files"):
                continue

            prefix = file.stem.split("_")[0]
            with file.open("r", encoding="utf-8") as hdf_file_list:
                hdf_files = filter(lambda x: len(x) > 0, map(str.strip, hdf_file_list.readlines()))
                union_files = list(str(sample_dir / rel_path) for rel_path in hdf_files)

            merge_files[prefix].extend(union_files)

    transforms = Compose(
        [
            NormalizeScale(),
            FPS(args.num_points, device=args.device),
            PosToFloat32()
        ]
    )

    for key in merge_files:
        out_path = out_dir / f"{key}.h5"
        HdfIO.union_and_compress(merge_files[key], str(
            out_path), transforms=transforms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="A pth to PartNet data root")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="A path to output dir with union files")
    parser.add_argument("--num_points", type=int, default=4096, required=False,
                        help="A number of points to sample for FPS")
    parser.add_argument("--device", default="cpy", type=str, required=False,
                        help="A device to process")

    args = parser.parse_args()

    main(args)
