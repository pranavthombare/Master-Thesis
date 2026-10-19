"""Download Labeled Faces in the Wild (LFW) and its evaluation protocol files.

Uses the same figshare mirrors and SHA-256 checksums as scikit-learn's ``fetch_lfw_pairs``.
Images are the original (non-funnelled) 250x250 JPEGs, so every pipeline in the study
runs its own face detector and alignment.

    uv run python research/download_lfw.py            # -> data/lfw/
"""

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

FILES = {
    "lfw.tgz": (
        "https://ndownloader.figshare.com/files/5976018",
        "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
    ),
    "pairs.txt": (
        "https://ndownloader.figshare.com/files/5976006",
        "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
    ),
    "pairsDevTrain.txt": (
        "https://ndownloader.figshare.com/files/5976012",
        "1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa",
    ),
    "pairsDevTest.txt": (
        "https://ndownloader.figshare.com/files/5976009",
        "7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, checksum: str, out: Path) -> Path:
    dest = out / name
    if dest.exists() and sha256(dest) == checksum:
        print(f"ok        {name}")
        return dest
    print(f"download  {name}")
    urllib.request.urlretrieve(url, dest)
    actual = sha256(dest)
    if actual != checksum:
        dest.unlink()
        raise RuntimeError(f"checksum mismatch for {name}: {actual}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/lfw"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, (url, checksum) in FILES.items():
        fetch(name, url, checksum, args.out)

    images = args.out / "lfw"
    if not images.is_dir():
        print("extract   lfw.tgz")
        with tarfile.open(args.out / "lfw.tgz") as tar:
            tar.extractall(args.out, filter="data")
    n = sum(1 for _ in images.glob("*/*.jpg"))
    print(f"{n} images in {images}")


if __name__ == "__main__":
    main()
