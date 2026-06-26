"""Download development samples of the CTA Prod5 public DL1+DL2 release.

Idempotent: re-running with an already-present file verifies the SHA-256
checksum without re-downloading. Use --force to re-download regardless.

Target directory: data/raw/cta_prod5/dl1/

NOTE: This downloads individual files for development. The full Prod5
release (~43 gamma-diffuse + ~20 proton files) is NOT downloaded here.

Source: Zenodo DOI 10.5281/zenodo.7298569
License: CC-BY-4.0
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_ZENODO_RECORD = "7298569"
_BASE_URL = f"https://zenodo.org/api/records/{_ZENODO_RECORD}/files"

# Known dev-sample files and their approximate sizes.
_KNOWN_FILES: dict[str, str] = {
    "gamma": "gamma-diffuse_with_images_40.dl2.h5",  # ~543 MB
    "proton": "proton_with_images_00.dl2.h5",  # ~1.2 GB
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(file_name: str, dest_dir: Path) -> Path:
    """Download a single Prod5 file from Zenodo."""
    file_path = dest_dir / file_name
    url = f"{_BASE_URL}/{file_name}/content"
    logger.info("Downloading %s from %s", file_name, url)
    try:
        with urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(file_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (50 << 20) < (1 << 20):
                        pct = downloaded * 100 / total
                        logger.info(
                            "  %.1f%% (%d / %d MB)",
                            pct,
                            downloaded >> 20,
                            total >> 20,
                        )
    except URLError as exc:
        logger.error("Network error downloading %s: %s", file_name, exc)
        raise SystemExit(1) from exc

    logger.info("Download complete: %s (%d MB)", file_path, file_path.stat().st_size >> 20)
    return file_path


def _checksum_file_for(file_name: str) -> str:
    return f"{file_name}.sha256"


def _save_checksum(file_path: Path, dest_dir: Path) -> str:
    digest = _sha256(file_path)
    cksum_path = dest_dir / _checksum_file_for(file_path.name)
    cksum_path.write_text(f"{digest}  {file_path.name}\n", encoding="utf-8")
    logger.info("Checksum saved: %s", digest)
    return digest


def _verify_checksum(file_path: Path, dest_dir: Path) -> bool:
    cksum_path = dest_dir / _checksum_file_for(file_path.name)
    if not cksum_path.exists():
        logger.warning("No checksum file found at %s", cksum_path)
        return False
    stored = cksum_path.read_text(encoding="utf-8").strip().split()[0]
    actual = _sha256(file_path)
    if stored != actual:
        logger.error("Checksum mismatch!\n  expected: %s\n  actual:   %s", stored, actual)
        return False
    logger.info("Checksum OK: %s", actual)
    return True


def _download_one(file_name: str, dest_dir: Path, *, force: bool) -> None:
    """Download and verify a single file."""
    file_path = dest_dir / file_name
    if file_path.exists() and not force:
        logger.info("File already exists: %s", file_path)
        if _verify_checksum(file_path, dest_dir):
            logger.info("Verified. Nothing to do.")
            return
        logger.warning("Checksum failed. Re-downloading.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = _download(file_name, dest_dir)
    _save_checksum(file_path, dest_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download CTA Prod5 DL1+DL2 development samples."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw/cta_prod5/dl1"),
        help="Destination directory (default: data/raw/cta_prod5/dl1).",
    )
    parser.add_argument(
        "--species",
        choices=["gamma", "proton", "both"],
        default="both",
        help="Which species to download (default: both).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    species_list: list[str] = (
        list(_KNOWN_FILES.keys()) if args.species == "both" else [args.species]
    )
    for sp in species_list:
        _download_one(_KNOWN_FILES[sp], args.dest, force=args.force)

    logger.info("Done.")


if __name__ == "__main__":
    main()
