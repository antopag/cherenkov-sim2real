# UCI MAGIC → CTA Prod5 DL1 Schema Mapping

This document records the column mapping used by
`src/cherenkov_sim2real/data/cta_prod5.py` to align CTA Prod5 DL1
Hillas parameters to the UCI MAGIC 10-feature schema.

## Mapping table

| # | UCI MAGIC name | Prod5 ctapipe column | Mapping quality | Notes |
|---|----------------|---------------------|-----------------|-------|
| 1 | LENGTH | `hillas_length` | **Exact** | Semi-major axis of the image ellipse |
| 2 | WIDTH | `hillas_width` | **Exact** | Semi-minor axis of the image ellipse |
| 3 | SIZE | `hillas_intensity` | **Exact** | Total integrated charge (photo-electrons) |
| 4 | CONC | `concentration_cog` | **Approximate** | UCI "CONC" = ratio of sum of two highest pixels to SIZE. ctapipe `concentration_cog` = fraction of SIZE in COG pixel. Semantically close but not identical. |
| 5 | CONC1 | `concentration_core` | **Approximate** | UCI "CONC1" = ratio of highest pixel to SIZE. ctapipe `concentration_core` = fraction of SIZE in 3 core pixels. Semantically close but not identical. |
| 6 | ASYM | `hillas_skewness` | **Approximate** | UCI "ASYM" = distance from highest-pixel to centre along major axis, signed. ctapipe `hillas_skewness` = 3rd moment along major axis. Both measure image asymmetry but with different definitions. |
| 7 | M3LONG | `hillas_kurtosis` | **Weak proxy** | UCI "M3LONG" = 3rd root of third moment along major axis. ctapipe has no direct M3LONG; `hillas_kurtosis` (4th moment) is the closest available shape parameter. **Not a clean mapping.** |
| 8 | M3TRANS | `hillas_psi` | **Weak proxy** | UCI "M3TRANS" = 3rd root of third moment along minor axis. ctapipe has no M3TRANS; `hillas_psi` (image orientation angle) is the closest available angular parameter. **Not a clean mapping.** |
| 9 | ALPHA | `hillas_phi` | **Approximate** | UCI "ALPHA" = angle between major axis and line source→centre. ctapipe `hillas_phi` = azimuthal angle of image centroid in camera frame. Related but not identical. |
| 10 | DIST | `hillas_r` | **Exact** | Distance from image centroid to camera centre |

## Summary

- **3 exact mappings**: LENGTH, WIDTH, SIZE, DIST (via `hillas_r`)
- **3 approximate mappings**: CONC, CONC1, ASYM, ALPHA
- **2 weak proxies**: M3LONG, M3TRANS

The weak proxies (M3LONG → kurtosis, M3TRANS → psi) are the most
concerning for domain adaptation. CORAL and other feature-alignment
methods will operate on these columns, so the semantic mismatch must be
documented in the paper.

## Implications for S2 and CORAL

When CORAL aligns CTA Prod5 features to UCI MAGIC features (or vice
versa), the M3LONG and M3TRANS columns will be aligned despite measuring
different physical quantities. This is a known limitation documented in
PLAN.md §3.6 (S2 trade-offs). The decomposition framework handles this:
D(S2) - D(S0) captures the combined effect of real-data idiosyncrasies
AND schema mismatch.

## ctapipe columns NOT used (available for future work)

- `hillas_fov_lon`, `hillas_fov_lat`: image centroid in FoV coordinates
- `hillas_length_uncertainty`, `hillas_width_uncertainty`: fit uncertainties
- `timing_*`: image timing parameters (slope, intercept, deviation)
- `leakage_*`: image leakage fractions
- `concentration_pixel`: single-pixel concentration
- `morphology_*`: island counting, pixel counting
- `intensity_*`: intensity distribution statistics
- `peak_time_*`: peak time distribution statistics
- `core_psi`: core orientation
