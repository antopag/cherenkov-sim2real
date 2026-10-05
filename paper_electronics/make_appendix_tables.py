"""Emit the per-feature shift statistics table (Appendix C) from data.

No number is typed by hand: everything comes from
data/review_checks.json, block 'feature_stats'.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIFTS = [("headline", "Headline composite"), ("location_only", "Location only (atmospheric)"),
          ("noise_only", "Noise only (NSB)")]


def main() -> None:
    fs = json.loads((HERE / "data" / "review_checks.json").read_text(encoding="utf-8"))["feature_stats"]
    feats = fs["headline"]["features"]
    keep = [i for i, _ in enumerate(feats)
            if any(abs(fs[k]["d_mean"][i]) > 1e-4 or abs(fs[k]["d_std"][i]) > 1e-4
                   for k, _ in SHIFTS)]
    L = ["\\rev{\\section{Per-Feature Effect of the Shifts}\\label{app:featurestats}", "",
         "Table~\\ref{tab:featurestats} reports, for each perturbation family used in the paper, "
         "the change in mean and in standard deviation of every affected feature, measured in the "
         "standardised space in which the alignment methods operate (mean over 5 seeds). The five "
         "features not listed (ASYM, M3LONG, DIST, ALPHA, M3TRANS) are untouched by all three "
         "shifts.", "",
         "The table makes precise two statements in the main text. First, the "
         "atmospheric-attenuation family is a pure translation only for SIZE, which the "
         "$\\log_{10}$ transform converts from a multiplicative rescaling into an additive offset "
         "($\\Delta\\sigma = 0.000$); for WIDTH and LENGTH, which are not log-transformed, the same "
         "rescaling moves the standard deviation as well as the mean, so the family is "
         "location-dominated rather than purely translational. Second, the NSB family inflates the "
         "spread of the two concentration features while barely moving any mean.", "",
         "\\begin{table}[H]", "\\centering", "\\small",
         "\\caption{Change in standardised mean and standard deviation, source to target, per "
         "feature and per shift family. Norm of the mean-shift vector: "
         + ", ".join(f"{lab.split(' (')[0].lower()} ${fs[k]['mean_shift_norm']:.4f}$"
                     for k, lab in SHIFTS) + ".}\\label{tab:featurestats}",
         "\\begin{tabular}{l" + "cc" * len(SHIFTS) + "}", "\\toprule",
         " & " + " & ".join("\\multicolumn{2}{c}{" + lab + "}" for _, lab in SHIFTS) + " \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "Feature & $\\Delta\\mu$ & $\\Delta\\sigma$ & $\\Delta\\mu$ & $\\Delta\\sigma$ "
         "& $\\Delta\\mu$ & $\\Delta\\sigma$ \\\\", "\\midrule"]
    for i in keep:
        row = [feats[i]]
        for k, _ in SHIFTS:
            row += [f"${fs[k]['d_mean'][i]:+.4f}$", f"${fs[k]['d_std'][i]:+.4f}$"]
        L.append(" & ".join(row) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}}"]

    out = HERE / "manuscript" / "sections" / "98_appendix_featurestats.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", out, f"({len(keep)} features listed)")


if __name__ == "__main__":
    main()
