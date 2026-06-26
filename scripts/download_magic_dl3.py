"""Download MAGIC DL3 PDR1 (Crab Nebula) from Zenodo.

Idempotent: re-running with already-present files verifies the SHA-256
checksum without re-downloading. Use --force to re-download.

Source: Zenodo DOI 10.5281/zenodo.11108474
License: CC-BY-4.0
Reference: Nigro et al. 2024 (arXiv:2409.18823)
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

_URL = "https://zenodo.org/api/records/11108474/files/magic_dl3_pdr1.zip/content"
_ZIP_NAME = "magic_dl3_pdr1.zip"
_CHECKSUM_FILE = "sha256.txt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download MAGIC DL3 PDR1 (Crab Nebula)."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dest", type=Path, default=Path("data/raw/magic_dl3"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dest_dir: Path = args.dest
    marker = dest_dir / ".download_complete"

    if marker.exists() and not args.force:
        logger.info("MAGIC DL3 PDR1 already downloaded at %s", dest_dir)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MAGIC DL3 PDR1 from %s (~46 MB)", _URL)
    try:
        with urlopen(_URL) as resp:
            data = resp.read()
    except URLError as exc:
        logger.error("Network error: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Extracting zip (%d bytes)...", len(data))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
        logger.info("Extracted %d files", len(zf.namelist()))

    # Save checksum of the zip
    zip_path = dest_dir / _ZIP_NAME
    zip_path.write_bytes(data)
    digest = _sha256(zip_path)
    (dest_dir / _CHECKSUM_FILE).write_text(
        f"{digest}  {_ZIP_NAME}\n", encoding="utf-8"
    )
    logger.info("Checksum: %s", digest)

    marker.write_text("ok")
    logger.info("Done. MAGIC DL3 PDR1 saved to %s", dest_dir)


if __name__ == "__main__":
    main()
