# CTA Prod5 Public Release — Inventory

Date: 2026-04-30. Research only; no data downloaded.

---

## 1. Canonical release entry points and DOIs

| Record | DOI | Description |
|--------|-----|-------------|
| CTAO IRFs — prod5 v0.1 | [10.5281/zenodo.5499840](https://doi.org/10.5281/zenodo.5499840) | Pre-computed IRFs for both sites, 3 zenith angles, 3 azimuth options. Published 2021-09-23. |
| CTAO Simulation Telescope Models — prod5 | [10.5281/zenodo.6218687](https://doi.org/10.5281/zenodo.6218687) | sim_telarray configs, CORSIKA inputs, execution scripts (prod5b). CC-BY-4.0. |
| CTAO DL1+DL2 Event Lists — prod5 | [10.5281/zenodo.7298569](https://doi.org/10.5281/zenodo.7298569) | DL1+DL2 for CTA-North Alpha only. Published 2022-12-23. |
| CTAO IRF Comparison: prod5 vs prod3b | [10.5281/zenodo.8050921](https://doi.org/10.5281/zenodo.8050921) | Technical note (PDF). Published 2023-06-17. |

Reference paper: Gueta (for the CTA Consortium), "The Cherenkov
Telescope Array: layout, design and performance", arXiv:2108.04512
(ICRC 2021). Cited by the DL2 Zenodo record.

---

## 2. Geographic sites and array layouts

Configuration: **Alpha** (the only layout in public Prod5 releases).

| Site | Location | LSTs | MSTs | SSTs | Total |
|------|----------|------|------|------|-------|
| CTA-North | Roque de los Muchachos, La Palma | 4 | 9 | 0 | 13 |
| CTA-South | Near Paranal, Chile | 0 | 14 | 37 | 51 |

Source: CTAO performance page (https://www.ctao.org/for-scientists/performance/),
Zenodo 5499840.

Omega configuration (larger layout, Prod3b) is **not** part of Prod5.

---

## 3. Perturbation dimensions in the public release

### 3a. NSB level

**Not available publicly.** All public IRFs and DL2 event lists use
nominal dark-sky NSB only. No Zenodo records mention multiple NSB
levels for Prod5. Moonlight/elevated-NSB runs likely exist internally
but are **not publicly released**.

### 3b. Atmospheric profile

**Not available publicly.** The sim_telarray config (Zenodo 6218687)
includes site atmosphere parameters, but only one atmospheric profile
per site is documented in the public release. No winter/summer or
MODTRAN profile variations exposed.

### 3c. Zenith angle

**Partially available.**
- IRFs: **20, 40, 60 deg** (both sites). Source: Zenodo 5499840.
- DL1+DL2 event lists: **20 deg only** (North site only). Source:
  Zenodo 7298569.

### 3d. Azimuth

- IRFs: **North-pointing, South-pointing, azimuth-averaged** at each
  zenith angle.
- DL2 event lists: **North and South** pointings at 20 deg zenith.

### 3e. Particle species (public DL2, North only, 20 deg zenith)

| Species | Public? | Statistics |
|---------|---------|------------|
| Gamma-diffuse | Yes | ~43 files, ~544-571 MB each (HDF5 with images) |
| Proton | Yes | ~20 files, ~1.18-1.22 GB each (HDF5 with images) |
| Electron | Not confirmed | |
| Gamma point-source | Not confirmed at DL1/DL2 | Used internally for IRF derivation |

---

## 4. Data levels exposed

| Data level | Public? | Details |
|------------|---------|---------|
| DL0 (raw simtel) | **No** | TB-scale; remains internal to Consortium. |
| DL1 (calibrated images + Hillas) | **Yes** | HDF5, CTA-North only, 20 deg zenith, gamma-diffuse + proton. Files labelled "with images". Source: Zenodo 7298569. |
| DL2 (reconstructed events) | **Yes** | HDF5, same conditions as DL1. Consolidated files ~1.5 GB. **Caveat:** record states "this repository only contains the direction information" — energy reconstruction may be incomplete. Source: Zenodo 7298569. |
| IRFs | **Yes** | FITS and ROOT formats. Both sites, 3 zenith angles, 3 azimuth options. Full archive 924 MB; FITS-only 44 MB. Source: Zenodo 5499840. |

All are openly downloadable from Zenodo without authentication.

**Requires Consortium membership:** DL0/simtel files, runs at
additional zenith/NSB/atmosphere conditions, CTA-South DL1/DL2,
gamma point-source and electron DL1/DL2.

---

## 5. File formats and tooling

| Format | Used for | Details |
|--------|----------|---------|
| FITS | IRFs | Standard GADF tables. 0.2 dex log-energy binning. |
| ROOT | IRFs | Alternative format. |
| HDF5 (.h5) | DL1 + DL2 event lists | ctapipe-native schema. |
| .tar.gz | sim_telarray configs | ~590 KB config, ~36 KB examples. |

Software versions used to produce the public files:
- **ctapipe 0.17** (DL1/DL2 production).
- **CORSIKA 7.71** with QGSjet-II-04 + URQMD.
- **sim_telarray:** version embedded in config tarball (prod5b).

Readers: **ctapipe >= 0.17**, **pyeventio** (for simtel; not needed
for HDF5). Source: Zenodo 7298569.

File sizes:
- Gamma-diffuse with images: ~544-571 MB per file, 43 files.
- Proton with images: ~1.18-1.22 GB per file, 20 files.
- Consolidated gamma-diffuse (no images): ~1.5 GB.
- Consolidated proton (no images): ~1.55 GB.
- Total record: 74 files.

---

## 6. License, citation, and acknowledgement

**License:** CC-BY-4.0 for all public Zenodo records.

**Required acknowledgement for IRFs:**
> "This research has made use of the CTA instrument response functions
> provided by the CTA Consortium and Observatory, see
> https://www.ctao-observatory.org/science/cta-performance/ (version
> prod5 v0.1; DOI 10.5281/zenodo.5499840) for more details."

**Required acknowledgement for DL1/DL2 event lists:**
> "This research has made use of the CTA DL1 and DL2 Event lists
> provided by the CTA Observatory and Consortium (version
> prod5-DL2-release1-DL2)"

---

## 7. Feasibility assessment for PLAN.md scenarios

### S0 (controlled MC/MC — clean reference)

**NOT FEASIBLE with public Prod5 event-level data as currently
released.** The public DL1/DL2 event lists are at a single zenith
(20 deg), single NSB (dark), single atmospheric profile, and single
site (North). There are no multi-condition event lists to construct
a source-target pair.

**Partial workaround — azimuth split:** DL2 includes North-pointing
and South-pointing runs at 20 deg zenith. Azimuth-induced shift is
weak for IACTs (mainly affects geomagnetic field effects on shower
development). This would produce a very small domain shift —
potentially too small to be a meaningful DA benchmark.

**Alternative — re-simulation:** The public sim_telarray configs
(Zenodo 6218687) are CC-BY-4.0 and could be used to re-simulate at
different zenith angles / NSB levels using CORSIKA + sim_telarray
locally. This is computationally expensive (days to weeks on a single
workstation) but technically feasible. Requires CORSIKA license
(free for academic use) and sim_telarray (public).

### S2 (CTA Prod5 single-tel subset -> MAGIC DL3)

**Feasible with caveats.** The public DL1 data includes per-telescope
Hillas parameters for all telescopes in the Alpha-North array (4 LSTs
+ 9 MSTs). A single-LST subset can be extracted by selecting events
triggered by a single LST and using only that telescope's Hillas
parameters. Statistics: the gamma-diffuse dataset has ~43 files;
total event count depends on LST trigger rate but should be
substantial (tens of thousands to hundreds of thousands of events).

**Caveat:** The DL2 energy/direction reconstruction used array-level
stereo information even for events that triggered a single LST. To
get a true single-telescope equivalent, one would need to use the
DL1-level per-telescope parameters only, not the DL2 stereo-
reconstructed quantities.

### S3 (CTA Prod5 full -> MAGIC DL3)

**Feasible.** Use the full CTA-North Alpha DL1/DL2 dataset at 20 deg
zenith. Natural choice: Alpha-North configuration (4 LST + 9 MST).

### Img-A (image sim-to-sim controlled)

**Same constraint as S0.** Requires multi-condition image data.
Public release has images at one condition only (20 deg zenith, dark
NSB, North site).

---

## 8. Caveats and known issues

1. **Preliminary status.** DL2 record (Zenodo 7298569) explicitly
   warns: "Products are preliminary and don't reflect final CTA
   Observatory performance; data structures remain subject to change."
2. **Incomplete DL2.** Record states it "only contains the direction
   information" — energy estimates may not be fully included.
3. **North-only DL1/DL2.** No public DL1/DL2 for CTA-South.
4. **Single zenith in DL1/DL2.** 20 deg only, despite IRFs covering
   20/40/60 deg.
5. **No DL0.** Raw simtel files not publicly released.
6. **Prod5 supersedes Prod3b.** Comparison note (Zenodo 8050921)
   recommends Prod5 over Prod3b.
7. **Prod6 exists.** Prod6 telescope models published (Zenodo
   14198379); Prod5 may eventually be superseded.

---

## 9. Recommendation

The public CTA Prod5 release is usable for S2 and S3 (source-side MC
against MAGIC DL3 real target) via the DL1 event lists from
Zenodo 7298569 (CTA-North Alpha, 20 deg zenith, gamma-diffuse +
proton). A single-LST extraction from DL1 serves S2; the full
Alpha-North array serves S3.

**S0 and Img-A are not feasible from the public release alone.**
The event-level data exists at a single observing condition. Options:
(a) re-simulate at additional conditions using the public
sim_telarray configs, (b) use the azimuth North/South split as a weak
controlled perturbation, or (c) redesign S0 to use a different
perturbation that can be applied synthetically (e.g. synthetic noise
injection, feature-space perturbation). This requires a PLAN.md
revision.

---

## Sources

All claims sourced from:
- https://doi.org/10.5281/zenodo.5499840 (IRFs)
- https://doi.org/10.5281/zenodo.6218687 (sim_telarray configs)
- https://doi.org/10.5281/zenodo.7298569 (DL1+DL2 event lists)
- https://doi.org/10.5281/zenodo.8050921 (IRF comparison note)
- https://www.ctao.org/for-scientists/performance/
- arXiv:2108.04512 (Gueta, ICRC 2021)
