import os
import argparse

from downloader_cli.download import Download


def main(args):
    if os.path.exists(args.out) and not args.force:
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Download(args.url).download()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="An url")
    parser.add_argument("--out", type=str, required=True, help="Output path")
    parser.add_argument("--force", action="store_true", required=False, help="Overwrite if exist")

    args = parser.parse_args()

    main(args)
