# CLAUDE.md — guide for contributors and AI assistants

This file is the operating manual for any human or AI assistant working on
**cherenkov-sim2real**. Read it before touching code.

The single source of truth for the **research plan** is
[PLAN.md](PLAN.md). Whenever the plan, the experimental matrix, the dataset
list, or the timeline changes, update PLAN.md in the same change.

---

## 1. Project goal

A reproducible benchmark of **sim-to-real domain adaptation** methods for
gamma/hadron separation and gamma-ray event classification with Imaging
Atmospheric Cherenkov Telescopes (IACTs).

- **Source domain:** Monte Carlo simulations.
- **Target domain:** real observations.
- **Output:** a paper for the MDPI journal *Electronics*, Special Issue
  *"Data-Related Challenges in Machine Learning: Theory and Application"*.
- **Deadline:** 30 November 2026.

---

## 2. Hard constraint — public data only

> **This project uses only publicly available, openly licensed datasets,
> simulations, and analysis tools.** No proprietary or collaboration-internal
> data, Monte Carlo productions, calibration files, or pipelines may enter the
> project, directly or as a transitive dependency.

This is non-negotiable. If a future change introduces a non-public dependency,
it must be reverted before merging. When in doubt, ask. This rule exists so
that every result in the paper can be reproduced by anyone from the sources
listed in Section 3.

---

## 3. Allowed datasets

Use only the following (and only the public, openly licensed releases):

- **UCI MAGIC Gamma Telescope dataset** — tabular Hillas parameters.
  Lightweight, fast iteration, well-known baseline.
- **MAGIC public / open data releases** — when a public release is
  available, used as the "real observation" target on the tabular side
  (and image side if released that way).
- **CTA Prod5 public Monte Carlo simulations** — large MC source domain
  for both tabular and image representations.
- **(Optional)** **H.E.S.S. DL3 public release** — DL3-level real
  observations for cross-instrument shift studies.
- **(Optional)** **VERITAS public data** — only if it adds meaningful
  diversity beyond the above.

Anything not on this list requires a discussion and an update to PLAN.md
before being introduced.

---

## 4. Environment

This project runs inside the conda environment **`cherenkov`** (not
`angelica` — `angelica` is a pre-existing environment for other projects
and must not be modified). Python **3.11.15**. All dependencies are
declared in **`environment.yml`**. Do not introduce poetry, uv, or
pipenv. The package itself is installed in editable mode with
`pip install -e .` inside the conda env.

Practically:

```bash
# First time only, if cherenkov does not yet exist:
conda env create -f environment.yml

# Or, to update an existing cherenkov with anything new:
conda env update -n cherenkov -f environment.yml --prune

conda activate cherenkov
pip install -e .
```

`environment.yml` is the **single source of truth** for runtime and dev
dependencies. `pyproject.toml` only configures ruff, black, mypy, and
pytest, plus the editable-install metadata. Do not move runtime deps into
`pyproject.toml`.

**Key version pins** (rationale in PLAN.md §3.2 and §3.3):
- `gammapy=1.1` — validated against MAGIC PDR1.
- `ctapipe=0.17.0` — schema match for CTA Prod5 DL1/DL2.
- `numpy<2.0`, `astropy<6.0` — required by gammapy 1.1.
- `setuptools<81` — gammapy 1.1 uses deprecated `pkg_resources`.

**GPU notes.** Torch is currently **CPU-only** (pip-installed,
`torch 2.11.0+cpu`). GPU build is deferred until image experiments begin
(estimated session 7+). When activating GPU support, the expected
sequence is:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Single-env approach (`cherenkov`) is preferred. If the CUDA-enabled torch
creates conflicts with the existing gammapy/ctapipe stack, fall back to a
sub-environment **`cherenkov-gpu`**.

**Rejected alternative.** A two-env split (`cherenkov` + `cherenkov-gpy`
for Gammapy isolation) was considered and rejected — single-env works.

**BLAS: OpenBLAS only, no MKL.** The conda-forge MKL build
(`libblas=*=*mkl*`, MKL 2025.3) caused silent segfaults on matrix
operations with more than ~100 rows on Windows. The environment pins
`libblas=*=*openblas*` and `nomkl` to force OpenBLAS. **Do not relax
these pins.** If a future dependency forces MKL, isolate it in a
sub-environment rather than removing the OpenBLAS pin.

**Known platform limitations.**
- **ctapipe.io is unusable on Windows** because of a transitive
  dependency on the Unix-only `pwd` module (`ctapipe.io.metadata`
  imports `pwd`). The Prod5 loader
  (`src/cherenkov_sim2real/data/cta_prod5.py`) reads HDF5 directly via
  PyTables to bypass this. Consequence: DL0 / simtel raw access is not
  possible on Windows. If raw simtel reading becomes necessary in the
  future, switch to Linux or WSL. This is not a blocker for the paper,
  which uses DL1/DL2 only.

---

## 4b. Hydra output convention

All experiment scripts use Hydra for configuration. The output directory
pattern is:

```yaml
hydra:
  run:
    dir: results/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: results/${experiment_name}/multirun_${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
```

**`experiment_name` is a REQUIRED field** in every Hydra config under
`configs/experiment/`. It has no default value and no fallback — Hydra
must error out clearly if it is missing. This is a project rule.

**The default Hydra `outputs/` directory at the project root must NOT be
used.** Any experiment script that writes to `outputs/` is a bug. If you
see an `outputs/` directory in the project root, investigate and fix the
script that created it.

---

## 5. Reproducibility rules

1. **Seeds.** Every experiment runs with `N` seeds (default `N=5`).
   Report mean ± std. Single-seed numbers are not allowed in the paper.
2. **Configuration.** All experiments are configured via Hydra YAML files
   under `configs/`. CLI flags exist only for `run_experiment.py` and only
   to select / override Hydra configs.
3. **Tracking.** Each run is logged to MLflow (offline mode is fine; the
   `mlruns/` directory stays out of git). The MLflow run id is also
   written into the per-experiment manifest (see below).
4. **Data.** Data is downloaded by `scripts/download_*.py` into `data/`
   (gitignored). No dataset, raw or derived, is ever committed.
5. **Results.** Each experiment writes to
   `results/<experiment_name>/<timestamp>_<seed>/` and emits a
   `manifest.json` with: config snapshot, git SHA, conda env name, package
   versions, random seeds, dataset hashes, MLflow run id, and final
   metrics. The manifest is the artifact the paper cites.
6. **Notebooks.** Notebooks are for exploration and figure rendering only.
   No notebook is part of the reproducibility chain. If a result needs to
   be reproduced, it must come from a script, not a notebook.

---

## 6. Coding style

- Python 3.11+.
- **Type hints everywhere**, including private helpers and tests.
- `ruff` + `black` + `mypy --strict` must pass before any merge.
- `pytest` for unit tests. Aim for tests on:
  - data loaders (shape, dtype, label distribution),
  - metrics (ML and physics),
  - shift estimators (MMD, A-distance, calibration drift).
- Keep modules small and single-purpose. Public API of each subpackage is
  exposed in its `__init__.py`.
- Do not import from sibling subpackages across layer boundaries except
  through documented entry points (e.g. `training/` may import from
  `models/`, `data/`, `adaptation/`, `metrics/`; `data/` must not import
  from `models/`).

---

## 7. Directory layout

```
cherenkov-sim2real/
├── README.md                    # short project description
├── CLAUDE.md                    # THIS FILE — guide for humans + AI
├── PLAN.md                      # research plan (single source of truth)
├── environment.yml              # conda env `cherenkov` — runtime + dev deps
├── pyproject.toml               # editable install + ruff/black/mypy/pytest
├── .gitignore
├── .python-version              # informational; conda env is authoritative
├── configs/                     # Hydra configs
│   ├── data/                    # one yaml per dataset / split scheme
│   ├── model/                   # one yaml per model architecture
│   ├── adaptation/              # one yaml per adaptation method
│   └── experiment/              # composed configs for full experiments
├── src/cherenkov_sim2real/
│   ├── data/                    # dataset loaders & splits, no model code
│   │   ├── magic_uci.py         # UCI MAGIC tabular loader
│   │   ├── magic_open.py        # MAGIC public-release loader
│   │   └── cta_prod5.py         # CTA Prod5 MC loader
│   ├── models/                  # architectures only, no training loops
│   │   ├── tabular.py           # MLP / GBDT-style heads on Hillas params
│   │   └── image.py             # CNN / ViT on Cherenkov shower images
│   ├── adaptation/              # domain-adaptation methods
│   │   ├── coral.py
│   │   ├── mmd.py
│   │   ├── dann.py
│   │   ├── self_training.py
│   │   └── tta.py               # test-time adaptation: TENT, BN-only, ...
│   ├── shift/                   # domain-shift quantification
│   │   ├── mmd.py               # kernel MMD on features
│   │   ├── a_distance.py        # proxy-A-distance via domain classifier
│   │   └── calibration.py       # ECE/Brier drift, reliability diagrams
│   ├── metrics/
│   │   ├── ml.py                # F1 macro, AUC, ECE, Brier
│   │   └── physics.py           # Q-factor, Li & Ma significance, eff/rej
│   ├── training/
│   │   └── trainer.py           # generic trainer; method-agnostic
│   └── reporting/
│       └── figures.py           # paper-quality plots
├── scripts/
│   ├── download_magic_uci.py
│   ├── download_cta_prod5.py
│   ├── run_experiment.py        # Hydra entry point
│   └── make_figures.py
├── tests/
├── notebooks/                   # exploration only
├── data/      (gitignored)
└── results/   (gitignored)
```

Module roles in one line each:

- `data/` — turn raw files on disk into `(X, y)` tensors / dataframes
  plus split metadata. No models, no training.
- `models/` — pure `nn.Module` / sklearn-compatible classes. No I/O.
- `adaptation/` — domain-adaptation algorithms; each exposes a uniform
  `adapt(model, source_loader, target_loader, cfg)` interface.
- `shift/` — quantitative measurements of source↔target divergence.
- `metrics/` — ML metrics (`ml.py`) and physics metrics (`physics.py`),
  both with the same call signature `metric(y_true, y_pred, **kw)`.
- `training/` — a single configurable trainer used by all methods.
- `reporting/` — turn `manifest.json` files into figures and tables.

---

## 8. Naming conventions

### Experiment names

`<dataset_pair>__<representation>__<method>__<tag>`

- `dataset_pair` — `mc-uci` (MC source, UCI MAGIC target),
  `cta5-magic` (CTA Prod5 source, MAGIC public target),
  `cta5-hess` (CTA Prod5 source, H.E.S.S. DL3 target), etc.
- `representation` — `tab` (tabular Hillas) or `img` (shower images).
- `method` — `srconly`, `tgtonly`, `finetune`, `coral`, `mmd`, `dann`,
  `selftrain`, `tent`, `bnadapt`.
- `tag` — short free-form, e.g. `seed5` or `ablation-lambda`.

Examples:
- `mc-uci__tab__dann__seed5`
- `cta5-magic__img__coral__lr1e3`

### Config files

- `configs/data/<dataset_pair>.yaml`
- `configs/model/<representation>_<arch>.yaml`
  (e.g. `tab_mlp.yaml`, `img_resnet18.yaml`)
- `configs/adaptation/<method>.yaml`
- `configs/experiment/<experiment_name>.yaml` — composes the others.

### Result directories

`results/<experiment_name>/<YYYYMMDD-HHMMSS>_seed<N>/`

Each contains `manifest.json`, `metrics.json`, `predictions.parquet`,
and any figures specific to that run.

---

## 9. Do / don't checklist for future sessions

**Do**
- [ ] Re-read PLAN.md at the start of every coding session.
- [ ] Update PLAN.md in the same change whenever the plan shifts.
- [ ] Add or update a config file before adding a new training pathway.
- [ ] Run `ruff check`, `black --check`, `mypy`, and `pytest` before
      declaring a task done.
- [ ] Use multiple seeds and report mean ± std.
- [ ] Write a unit test for any new metric, loader, or shift estimator.
- [ ] Cite the dataset version / Prod5 release in the manifest.

**Don't**
- [ ] Don't introduce non-public data, MC productions, or tooling — ever.
- [ ] Don't migrate to poetry / uv / pipenv. Conda env `cherenkov` +
      `environment.yml` is the convention.
- [ ] Don't put runtime dependencies in `pyproject.toml`.
- [ ] Don't commit anything under `data/`, `results/`, or `mlruns/`.
- [ ] Don't put critical logic in notebooks.
- [ ] Don't report single-seed results in the paper.
- [ ] Don't import across module boundaries in violation of section 6.
- [ ] Don't add a new dataset, method, or evaluation metric without
      first updating PLAN.md.

---

## 10. PLAN.md is the source of truth

The research plan — research questions, hypotheses, datasets, methods,
experimental matrix, evaluation protocol, risks, and timeline — lives in
[PLAN.md](PLAN.md). Treat it like a contract:

- Any code change that contradicts PLAN.md must update PLAN.md first.
- Any new experiment must be representable in PLAN.md's experimental
  matrix.
- If PLAN.md and code disagree, PLAN.md wins until the disagreement is
  resolved.
