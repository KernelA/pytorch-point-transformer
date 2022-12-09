import os
import argparse

import gdown

def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gdown.download(args.url, output=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="A gdrive file id")
    parser.add_argument("--out", type=str, required=True, help="Output name")

    args = parser.parse_args()

    main(args)
