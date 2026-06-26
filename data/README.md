# data/

This directory is **gitignored**. Datasets are downloaded here by the
scripts under `../scripts/download_*.py`. Never commit anything from
this folder. See CLAUDE.md §5.

Expected layout (created on first download):

    data/
    ├── raw/
    │   └── uci_magic/
    │       ├── magic04.data     # raw CSV, ~19k events, 10 Hillas features + label
    │       └── sha256.txt       # checksum computed on first download
    ├── magic_open/              # MAGIC PDR1 DL3 (future)
    ├── cta_prod5/
    │   └── dl1/
    │       ├── gamma-diffuse_with_images_40.dl2.h5   # dev sample only (~543 MB)
    │       └── sha256.txt
    ├── hess_dl3/                # optional
    └── veritas/                 # optional

The UCI MAGIC dataset is Monte Carlo (not real observations).
See PLAN.md S3.1 for the clarification.

The CTA Prod5 development sample is a **single gamma-diffuse file** from
the full Prod5 release (Zenodo DOI 10.5281/zenodo.7298569). It contains
only gamma events (no protons). The full download (43 gamma + 20 proton
files, ~50 GB total) is deferred to a later session. CC-BY-4.0.
