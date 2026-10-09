# cherenkov-sim2real

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23257260.svg)](https://doi.org/10.5281/zenodo.23257260)

Reproducible benchmark of **feature-alignment domain adaptation** under
distribution shift, measured across five carrier classifiers, on tabular
Imaging Atmospheric Cherenkov Telescope (IACT) features.

What this repository contains is the **carrier–DA interaction benchmark**: a
controlled shift of **tabular Hillas parameters** from public CTA Prod5 Monte
Carlo, five carriers (logistic regression, shallow MLP, LightGBM, ExtraTrees,
random forest) crossed with source-only, mean matching, CORAL and RBF-kernel
MMD, fifteen seeds per cell, plus ablations over tree depth, shift type and
shift intensity. Only **publicly available** data are used.

The wider programme this repository was scaffolded for — adversarial adaptation
(DANN), self-training, test-time adaptation, the Cherenkov shower **image** arm,
and a quantitative MC→real pair — is **not** part of the published study and is
future work. Some of that machinery exists in `src/` but is not wired into the
runner and produced no result reported in the paper. A quantitative MC→real
benchmark in Hillas space is not realisable on currently public data: MAGIC DL3
exposes only reconstructed energy, with no Hillas parameters.

This work accompanies the paper *"When Feature Alignment Helps, Hurts, or Washes Out: Carrier-Dependent Domain Adaptation for Cherenkov Telescope Event Classification"*, accepted in *Algorithms* (MDPI), manuscript algorithms-4602126.
The archived release for that paper is **v1.0.0**, DOI
[10.5281/zenodo.23257260](https://doi.org/10.5281/zenodo.23257260).

- See **[CLAUDE.md](CLAUDE.md)** for contributor and AI-assistant guidelines,
  reproducibility rules, and the project's conventions.

## Evaluation protocol (read this first)

The target domain of this benchmark is a perturbed copy of the source
sample. If a carrier is trained on 70% of that sample and then scored on
the *whole* perturbed copy, most of its "target" is a perturbed copy of
its own training set. High-capacity carriers memorise those events: an
unbounded-depth Random Forest scores AUC 1.000 on its training events and
0.766 on events it has never seen. Scoring that way inflates the measured
benefit of domain adaptation by a factor that tracks memorisation
capacity, from 1.0x for logistic regression to 5.7x for that Random
Forest, which manufactures a carrier gradient out of a capacity gradient.

**Every number in the paper is therefore computed on the held-out
target**: the perturbed copies of the 30% of events excluded from
training. `scripts/run_heldout_matrix.py` implements this; the earlier
full-target path is kept only to quantify the inflation (Figure 10 and
Table 2 of the paper). If you reuse this benchmark, keep the split.

## Reproducing the paper

The consolidated results are shipped as JSON, so every table and figure
regenerates without re-running any training:

```bash
conda env create -f environment.yml && conda activate cherenkov
pip install -e .
python paper_electronics/build_paper_data.py               # heldout.json -> paper_data.json
cd paper_electronics/figures && for f in generate_fig*.py; do python "$f"; done
```

To re-run the experiments, download the CTA Prod5 public files with
`scripts/download_cta_prod5.py`, then:

```bash
python scripts/run_heldout_matrix.py --which all    # main matrix, depth, shift, Q-vs-threshold
python scripts/run_memorisation_check.py            # the four-way memorisation diagnostic
python scripts/run_revision2.py --which sens        # DA-free marginal-sensitivity index
```

`scripts/run_experiment.py` and `scripts/run_ablations.py` produce the
original full-target runs, whose per-seed manifests are shipped in
`paper_electronics/data/manifests/` (991 files) and consolidated by
`extract_paper_data.py` and `extract_ablations.py`. They are retained
because the paper reports the comparison between the two protocols; they
are not the numbers the paper reports. The manuscript sources are not
part of this repository.

## Citation

See `CITATION.cff`. Please cite the paper for the results, and the archive for
the code:

> Pagliaro, A. *cherenkov-sim2real: code, configurations and per-seed results
> for "When Feature Alignment Helps, Hurts, or Washes Out: Carrier-Dependent
> Domain Adaptation for Cherenkov Telescope Event Classification"*, v1.0.0,
> Zenodo, 2026. DOI [10.5281/zenodo.23257260](https://doi.org/10.5281/zenodo.23257260)
