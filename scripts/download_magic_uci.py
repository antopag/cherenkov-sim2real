"""Download the UCI MAGIC Gamma Telescope dataset.

Idempotent: re-running with an already-present file verifies the SHA-256
checksum without re-downloading. Use --force to re-download regardless.

Target directory: data/raw/uci_magic/
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_UCI_URL = "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip"
_CHECKSUM_FILE = "sha256.txt"
_RAW_CSV = "magic04.data"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(dest_dir: Path) -> Path:
    """Download and extract the UCI MAGIC zip archive."""
    logger.info("Downloading UCI MAGIC dataset from %s", _UCI_URL)
    try:
        with urlopen(_UCI_URL) as resp:
            data = resp.read()
    except URLError as exc:
        logger.error("Network error downloading UCI MAGIC: %s", exc)
        raise SystemExit(1) from exc

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        logger.info("Zip contains: %s", names)
        zf.extractall(dest_dir)

    csv_path = dest_dir / _RAW_CSV
    if not csv_path.exists():
        # The CSV may be nested inside a subdirectory in the zip
        for name in names:
            if name.endswith(_RAW_CSV):
                csv_path = dest_dir / name
                break
    if not csv_path.exists():
        logger.error(
            "Expected file %s not found in zip. Contents: %s", _RAW_CSV, names
        )
        raise SystemExit(1)

    return csv_path


def _save_checksum(csv_path: Path, dest_dir: Path) -> str:
    digest = _sha256(csv_path)
    checksum_path = dest_dir / _CHECKSUM_FILE
    checksum_path.write_text(f"{digest}  {csv_path.name}\n", encoding="utf-8")
    logger.info("Checksum saved: %s", digest)
    return digest


def _verify_checksum(csv_path: Path, dest_dir: Path) -> bool:
    checksum_path = dest_dir / _CHECKSUM_FILE
    if not checksum_path.exists():
        logger.warning("No checksum file found at %s", checksum_path)
        return False
    stored = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    actual = _sha256(csv_path)
    if stored != actual:
        logger.error(
            "Checksum mismatch!\n  expected: %s\n  actual:   %s", stored, actual
        )
        return False
    logger.info("Checksum OK: %s", actual)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download the UCI MAGIC Gamma Telescope dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw/uci_magic"),
        help="Destination directory (default: data/raw/uci_magic).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dest_dir: Path = args.dest
    csv_path = dest_dir / _RAW_CSV

    if csv_path.exists() and not args.force:
        logger.info("File already exists: %s", csv_path)
        if _verify_checksum(csv_path, dest_dir):
            logger.info("Dataset is present and verified. Nothing to do.")
            return
        logger.warning("Checksum verification failed. Re-downloading.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _download(dest_dir)
    _save_checksum(csv_path, dest_dir)
    logger.info("Done. Dataset saved to %s", dest_dir)


if __name__ == "__main__":
    main()
