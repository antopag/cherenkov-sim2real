# cherenkov-sim2real

Reproducible benchmark of **sim-to-real domain adaptation** methods for
gamma/hadron separation and gamma-ray event classification with Imaging
Atmospheric Cherenkov Telescopes (IACTs).

The source domain is Monte Carlo simulation; the target domain is real
observations. We compare baselines, feature alignment (CORAL, MMD),
adversarial adaptation (DANN), self-training, and test-time adaptation
across both **tabular Hillas parameters** and **Cherenkov shower images**,
using only **publicly available** datasets (UCI MAGIC, MAGIC public
releases, CTA Prod5, optionally H.E.S.S. DL3 / VERITAS).

This work is being prepared for submission to the MDPI journal *Electronics*,
Special Issue *"Data-Related Challenges in Machine Learning: Theory and
Application"* (deadline 30 November 2026).

- See **[PLAN.md](PLAN.md)** for the research plan, datasets, methods,
  experimental matrix, and timeline.
- See **[CLAUDE.md](CLAUDE.md)** for contributor and AI-assistant guidelines,
  reproducibility rules, and the project's hard constraints.

> **Hard constraint:** this project does **not** use non-public data, non-public
> simulations, or non-public-derived pipelines. See CLAUDE.md.

## Reproducing the paper

All numbers, tables and figures of the paper regenerate from the
per-seed result manifests shipped in `paper_electronics/data/manifests/`
(one JSON per experiment and seed; 991 files) without re-running any
training:

```bash
conda env create -f environment.yml && conda activate cherenkov
pip install -e .
python paper_electronics/extract_paper_data.py      # main S0 matrix -> data/extracted_results.json
python paper_electronics/extract_ablations.py       # depth + shift ablations -> data/ablations.json
cd paper_electronics/figures && for f in generate_fig*.py; do python "$f"; done
```

To re-run the experiments themselves, download the CTA Prod5 public
files with `scripts/download_cta_prod5.py`, then use
`scripts/run_experiment.py experiment=<name>` for the main matrix
(configs in `configs/experiment/`) and `scripts/run_ablations.py` for the
two revision ablations. The MDPI LaTeX class is not redistributed here;
fetch it from <https://www.mdpi.com/authors/latex> into
`paper_electronics/manuscript/Definitions/` before compiling.

## Citation

See `CITATION.cff`. The archived release of this repository is deposited
on Zenodo (DOI added at publication).
