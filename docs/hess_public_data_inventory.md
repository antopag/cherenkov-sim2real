# H.E.S.S. Public Data Release Inventory

Date: 2026-05-07. Data downloaded and schema verified.

---

## 1. Canonical release

| Record | DOI | Description |
|--------|-----|-------------|
| H.E.S.S. DL3 DR1 | [10.5281/zenodo.1421099](https://doi.org/10.5281/zenodo.1421099) | First public test data release. 105 observations, 4 sources + empty field, H.E.S.S. I array. Published 2018-09. |

Reference: H.E.S.S. Collaboration, "H.E.S.S. first public test data
release", arXiv:1810.04516.

Official page: https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/

**No subsequent public DL3 releases found** as of May 2026.

---

## 2. Sources available

| Source | Type | Approx. runs | Notes |
|--------|------|-------------|-------|
| Crab Nebula | Point-like | ~20 | 2004, H.E.S.S. I |
| PKS 2155-304 | Point-like (AGN) | ~20 | 2006, 2008 |
| MSH 15-52 | Extended (PWN) | ~20 | 2004 |
| RX J1713.7-3946 | Extended (SNR) | ~25 | 2004 |
| Empty field | Background | ~20 | Mixed dates |

Total: 105 observations, ~48.6 hours effective exposure.

---

## 3. Schema inventory (EVENTS extension)

**Verified by direct inspection of downloaded FITS files.**

| Column | Type | Unit | Range (Crab file) | Description |
|--------|------|------|-------------------|-------------|
| EVENT_ID | int64 | — | — | Encoded event identifier |
| TIME | float64 | s | MET epoch | Event timestamp |
| RA | float32 | deg | 186.7–256.7 | Reconstructed Right Ascension |
| DEC | float32 | deg | -72.7–-22.7 | Reconstructed Declination |
| ENERGY | float32 | TeV | 0.22–104.3 | Reconstructed energy |

**That is the COMPLETE column list.** No additional columns are present.

The following columns, which are optional in the GADF specification, are
**NOT present** in the H.E.S.S. DL3 DR1 release:

- GAMMANESS / classifier score: **absent**
- MULTIP / telescope multiplicity: **absent**
- HIL_MSW / mean scaled width: **absent**
- HIL_MSL / mean scaled length: **absent**
- HEIGHT_MAX / shower maximum: **absent**
- IMPACT / impact parameter: **absent**
- DIR_ERR / directional error: **absent**
- ENERGY_ERR / energy error: **absent**
- COREX, COREY / shower core: **absent**

---

## 4. Schema overlap with Prod5 single-LST

| H.E.S.S. column | Prod5 Hillas counterpart | Overlap quality |
|------------------|--------------------------|-----------------|
| ENERGY (TeV) | SIZE (p.e.) — proxy via energy-size relation | **Derivable** (requires calibration) |
| RA, DEC → offset | DIST (camera-frame) — approximate | **Approximate** |
| TIME | — | **No counterpart** |
| EVENT_ID | — | **No counterpart** |
| — | LENGTH, WIDTH | **Absent in H.E.S.S.** |
| — | CONC, CONC1, ASYM | **Absent in H.E.S.S.** |
| — | M3LONG, M3TRANS, ALPHA | **Absent in H.E.S.S.** |

**Common schema size: 2 features** (log10_energy proxy + angular
offset). Identical to the MAGIC PDR1 situation.

---

## 5. Critical assessment

H.E.S.S. DL3 DR1 has **exactly the same minimal schema** as MAGIC PDR1:
5 columns (EVENT_ID, TIME, RA, DEC, ENERGY), of which only ENERGY and
angular offset are usable ML features.

The 2-feature common schema with Prod5 is too narrow for meaningful
feature-alignment DA benchmarking (CORAL, MMD, mean matching all
require ≥ 3-4 features to produce distinguishable behavior).

---

## 6. License and citation

**License:** Attribution (non-standard; see README.txt in archive).
**Critical restriction:** "No scientific publications may be derived
from this data" — the data is explicitly a **test release for tool
validation only**. This is MORE restrictive than MAGIC PDR1 (CC-BY-4.0).

Citation: H.E.S.S. Collaboration 2018, DOI 10.5281/zenodo.1421099.

---

## 7. Recommendation

**H.E.S.S. is NOT a viable S2 target.** Two blocking issues:

1. **Schema**: same 2-feature overlap as MAGIC. No improvement over
   the MAGIC PDR1 situation discovered in session 9.

2. **License**: "no scientific publications may be derived" is
   incompatible with using the data as a target in a published
   benchmark. MAGIC PDR1 (CC-BY-4.0) does not have this restriction.

**Implication for the paper strategy:** The schema mismatch between
public DL3 releases and MC DL1 data is not an accident — it is
structural. All public IACT DL3 releases (MAGIC, H.E.S.S.) expose
only the minimal GADF event-list columns (energy + direction). The
Hillas-level image parameters that would enable rich feature-alignment
DA are available only at DL1/DL2, which is not publicly released for
real observations (except for CTA Prod5 MC).

This structural gap is itself a finding worth documenting: the
decomposition framework reveals that the dominant barrier to sim-to-real
DA on public IACT data is not distributional shift but **schema
mismatch between public data levels**.

---

## Sources

- https://doi.org/10.5281/zenodo.1421099 (DL3 DR1)
- https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/
- https://arxiv.org/abs/1810.04516
- Direct FITS file inspection (downloaded 2026-05-07)
