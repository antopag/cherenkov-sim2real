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

This work accompanies the paper *"When Feature Alignment Helps, Hurts, or Washes Out: Carrier-Dependent Domain Adaptation for Cherenkov Telescope Event Classification"*, accepted in *Algorithms* (MDPI), manuscript algorithms-4602126.

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

See `CITATION.cff`. The archived release of this repository is deposited
on Zenodo (DOI added at publication).
