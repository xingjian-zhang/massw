"""Data loading and processing utilities."""
import os
import sys

import wget

# Setting the project directory relative to this script's location
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_dataset(version="v1"):
    """Download the dataset from remote storage."""
    urls = {
        "v1": {
            "massw_metadata_v1.jsonl":
            "https://www.dropbox.com/scl/fi/r2jlil9lj0ypo2fpl3fxa/\
            massw_metadata_v1.jsonl?rlkey=ohnriak63x4ekyli25naajp0q&dl=1",
            "massw_v1.tsv":
            "https://www.dropbox.com/scl/fi/ykkrpf269fikuchy429l7/\
            massw_v1.tsv?rlkey=mssrbgz3k8adij1moxqtj34ie&dl=1",
        }
    }
    try:
        files = urls[version]
    except KeyError as e:
        raise ValueError(
            f"Invalid version: {version}.\
            Choose from {list(urls.keys())}") from e
    for filename, url in files.items():
        print(f"Downloading {filename}...")
        # Constructing the output path
        out_path = os.path.join(PROJECT_DIR, "data", filename)
        if os.path.exists(out_path):
            print(f"{filename} already exists. Skipping download.")
            continue
        wget.download(url, out=out_path, bar=bar_progress)


def bar_progress(current, total, width=80):
    """Display a progress bar for the download."""
    progress_message = f"Downloading: {current / total * 100:.0f}% \
                         [{current} / {total}] bytes"
    # Don't use print() as it will print in new line every time.
    width = min(width, 100)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


if __name__ == "__main__":
    download_dataset()
