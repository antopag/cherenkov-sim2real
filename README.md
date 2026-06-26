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
