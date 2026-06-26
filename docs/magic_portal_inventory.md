# MAGIC Public Data Portal — DL3 Inventory

Date: 2026-04-30. Research only; no data downloaded.

---

## 1. Portal URLs and canonical entry points

| Resource | URL |
|----------|-----|
| MAGIC Open Data Portal | https://opendata.magic.pic.es/ |
| Zenodo release (PDR1) | https://zenodo.org/records/11108474 |
| Zenodo MAGIC community | https://zenodo.org/communities/magictelescopes/records |
| Reference paper | Nigro et al. (2024), arXiv:2409.18823; published in JHEAP |
| GADF spec | https://gamma-astro-data-formats.readthedocs.io/ |

The only DL3-level public release is **PDR1** (Public Data Release 1),
hosted on Zenodo with DOI [10.5281/zenodo.11108474](https://doi.org/10.5281/zenodo.11108474).
The portal at opendata.magic.pic.es also hosts older, non-DL3 datasets
(see section 5).

---

## 2. Available sources in PDR1

### 2.1 Crab Nebula (the only public DL3 source)

| Parameter | Dark conditions | Moonlight (NSB 1-8) |
|-----------|----------------|---------------------|
| Runs | ~169 | ~120 |
| Effective exposure | ~42 h | ~20 h |
| Observation period | 2013-2018 | Nov 2018 - Sep 2019 |
| Zenith range | 5-50 deg | 5-50 deg |
| File size (DL3) | ~45 MB | ~14 MB |

**Total Crab: ~60 hours, ~289 runs.**

- **Data format:** FITS, GADF-compliant. One DL3 FITS file per
  observation run (~15-20 min each). Each file contains an event list
  HDU and IRF HDUs.
- **IRFs:** Point-like only (energy-dependent directional cut in
  `RAD_MAX_2D` table). Full-enclosure (3D) IRFs are NOT included.
  Single-offset (0.4 deg wobble) and multi-offset (0.2-1.4 deg) IRFs
  available.
- **IRF components:** Effective area, energy dispersion, PSF,
  RAD_MAX_2D.
- **Multi-offset binning:** 0.20, 0.35, 0.40, 0.70, 1.00, 1.40 deg.
- **Validated energy range:** 80 GeV - 20 TeV (instrument capability
  ~30 GeV - ~100 TeV).
- **GADF version:** No formal version number; conforms to the living
  GADF community spec as of 2024.
- **DOI:** 10.5281/zenodo.11108474 (Zenodo v0.2, May 2024).
- **License:** CC-BY-4.0.
- **Required attribution:** Credit to the MAGIC Collaboration; cite
  Nigro et al. (2024), arXiv:2409.18823.

### 2.2 Mrk 421

**Not publicly released.** The reference paper validated 42 h / 176
runs of Mrk 421 data (from 2014) internally, but this dataset is NOT
part of PDR1 and is NOT available on any public portal.

### 2.3 Mrk 501

**Not publicly released.** Not confirmed as even validated in the
reference paper.

### 2.4 Other sources validated internally (NOT public)

- IC 310: 3.5 h, 11 runs (Nov 2012 flare).
- QSO B0218+357: 2 h, 7 runs (Jul 2014).
- M15: 57 h, 200 runs (2015-2016).

None of these are part of PDR1.

### 2.5 Other datasets on opendata.magic.pic.es (NOT DL3)

These are older releases in custom/proprietary formats, predating
GADF standardisation:

- **MAGIC_2021RSOph** — RS Ophiuchi nova (released 2022-02-17).
- **MAGIC_JointCrab** — 2 Crab runs for stereo upgrade validation
  (released 2019-05-02). Used in gammapy tutorials; has partial DL3
  structure but is not part of PDR1.
- **MAGIC_TXS0506+056** — TXS 0506+056 blazar, Sep-Oct 2017 (released
  2018-07-12).

DL3/GADF compliance of these older releases is **not confirmed**.

---

## 3. Crab Nebula: >= 5 h post-cut effective exposure?

**YES.** The dark-conditions subset alone provides ~42 h across ~169
runs. Even after quality cuts this far exceeds the 5-hour threshold.
The full release (dark + moonlight) totals ~60 h.

---

## 4. Mrk 421: >= 5 h post-cut effective exposure?

**NOT APPLICABLE.** There is no public DL3 release of Mrk 421. The
PLAN.md fallback to Mrk 421 is not viable under current public data
constraints. If a fallback is needed, it would have to come from a
different instrument (e.g. H.E.S.S. DL3 DR1) or the PLAN.md fallback
clause should be removed.

---

## 5. Caveats

1. **Moonlight systematics.** Crab spectrum is "slightly
   underestimated" for NSB levels 3-8 relative to reference, despite
   tuned image cleaning. Source: arXiv:2409.18823.
2. **Flux pipeline discrepancies.** Run-wise integral fluxes can
   differ ~20% between MARS (proprietary) and Gammapy pipelines;
   reduces to <5% with weekly binning. Source: arXiv:2409.18823.
3. **Point-like IRFs only.** No full-enclosure IRFs; morphological/3D
   analysis not possible.
4. **Single source.** Despite validating 166 h across 6 sources, only
   Crab is public.
5. **Software version.** Validated with Gammapy v1.1; current Gammapy
   is v2.0. Minor IRF interpolation differences may arise.
6. **Stereo vs mono mismatch with UCI MAGIC.** The UCI MAGIC dataset
   is MC for a single MAGIC-I telescope (mono, pre-stereo upgrade).
   PDR1 data is from the stereo system (two telescopes, post-2009).
   The Hillas parameter space differs (mono vs stereo reconstruction).
   Additionally, UCI MAGIC operates at a lower analysis level than DL3.
   Feature-level alignment between UCI MAGIC 10-feature schema and
   DL3 event-list columns requires explicit schema mapping, documented
   in the loader.
7. **Zenodo version.** v0.2 as of May 2024; check for updates before
   download.

---

## 6. Recommendations

### For S1 (UCI MAGIC MC -> MAGIC DL3 tabular)

Use **PDR1 Crab Nebula dark-conditions subset** (~42 h, ~169 runs).

Justification: only substantial public DL3 dataset; dark conditions
minimise moonlight systematics; Crab is the canonical VHE standard
candle; CC-BY-4.0 license.

**Critical implementation note:** The feature-space mismatch between
UCI MAGIC (10 Hillas parameters from mono MC) and MAGIC DL3 (stereo
reconstructed event lists) is non-trivial. The DL3 event list columns
(reconstructed energy, direction, gammaness) are at a higher analysis
level than the UCI Hillas parameters. The S1 scenario requires either:
(a) projecting DL3 events into a Hillas-compatible feature vector
(if DL3 retains enough information), or (b) acknowledging that S1 is
a schema-mismatched scenario where the "same array" property refers
to the instrument, not the feature representation. This must be
resolved during loader implementation.

### For S6.6 (closing application)

Use the same PDR1 Crab dark subset. The Crab Nebula is the standard
reference source in VHE astronomy, directly comparable to the Crab
analysis in Pagliaro et al. (submitted) on non-public.

### Fallback

The PLAN.md fallback to Mrk 421 is **not viable** — no public DL3
data exists. Options:
(a) Remove the fallback clause (Crab statistics are more than
    sufficient).
(b) Replace Mrk 421 with H.E.S.S. DL3 DR1 data as a cross-instrument
    fallback (different instrument, different caveats).
(c) Keep the fallback clause but change it to "if Crab data quality
    is insufficient" rather than "if statistics are insufficient"
    (statistics are clearly sufficient).

---

## Sources

All claims sourced from:
- https://opendata.magic.pic.es/
- https://zenodo.org/records/11108474
- arXiv:2409.18823 (Nigro et al. 2024)
- https://docs.gammapy.org/2.0/tutorials/data/magic.html
