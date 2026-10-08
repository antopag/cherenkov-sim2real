"""Emit tables/tab1_full_results.tex directly from data/paper_data.json.

No number in the table is typed by hand. Column order follows the rest of
the paper: LR, MLP, LightGBM, ExtraTrees, RF.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = [("lr", "LR"), ("mlp", "MLP"), ("lgbm", "LightGBM"),
         ("et", "ExtraTrees"), ("rf", "RF")]
METHODS = [("srconly", "Source-only"), ("mean_matching", "Mean matching"),
           ("coral", "CORAL"), ("mmd_rbf", "MMD-rbf")]
R = "\\rev{"


def zero(v: float) -> float:
    """Collapse a value that is zero to within rounding onto exact zero."""
    return 0.0 if abs(v) < 5e-3 else v


def cells(fn):
    return " & ".join(fn(c) for c, _ in ORDER)


def main() -> None:
    d = json.loads((HERE / "data" / "paper_data.json").read_text(encoding="utf-8"))
    M, D = d["matrix"], d["damage"]
    L = []
    add = L.append

    def val(c, m, k, nd=3):
        e = M[c][m][k]
        return f"{e['mean']:.{nd}f} $\\pm$ {e['std']:.{nd}f}"

    def delta(c, m, k, nd=4):
        e = M[c][m][f"delta_{k}"]
        lo, hi = e["ci95"]
        star = "*" if lo <= 0 <= hi else ""
        sign = "$+$" if e["mean"] > 0 else ("$-$" if e["mean"] < 0 else "$\\phantom{+}$")
        return f"{sign}{abs(e['mean']):.{nd}f} ({(hi - lo) / 2 * 1000:.1f}){star}"

    add("\\begin{table}[H]")
    add("\\begin{adjustwidth}{-\\extralength}{0cm}")
    add("\\centering")
    add("\\small")
    add("\\caption{\\rev{S0 benchmark on the held-out target, 15 seeds per cell. Blocks, top to "
        "bottom: AUC; shift damage $D$ (AUC lost by the unadapted carrier, held-out source versus "
        "held-out target) and the fraction $R$ of it each method recovers; paired $\\Delta$AUC "
        "relative to source-only, with the half-width of the paired 95\\% $t$-interval in "
        "parentheses (units of $10^{-3}$; an asterisk marks intervals containing zero); macro-F1 at "
        "the default operating point and its paired $\\Delta$; Q-factor at the default operating "
        "point. $R$ is a ratio of means with a bootstrap interval (Section~\\ref{sec:scenario}), reported for every cell in Table~\\ref{tab:recovery_ci}; "
        "the MLP entries are undetermined here because its damage is too small to divide by. The "
        "ranking metric and the two fixed-threshold metrics disagree on the linear carrier, where "
        "alignment changes the ranking only slightly but moves the score offset; "
        "Section~\\ref{sec:operating_points} resolves the disagreement.}}\\label{tab:full_results}")
    add("{\\revon")
    add("\\begin{tabular}{lccccc}")
    add("\\toprule")
    add(" & " + " & ".join(lab for _, lab in ORDER) + " \\\\")
    add("\\midrule")

    add("\\multicolumn{6}{l}{\\textit{AUC}} \\\\")
    for m, lab in METHODS:
        add(f"{lab:14s} & " + cells(lambda c, m=m: val(c, m, "auc")) + " \\\\")

    add("\\midrule")
    add("\\multicolumn{6}{l}{" + R + "\\textit{Shift damage $D$ and recovered fraction $R$}}} \\\\")
    add(R + "Damage $D$} & " + cells(lambda c: R + f"{D[c]['mean']:.4f}" + "}") + " \\\\")
    for m, lab in METHODS[1:]:
        def rec(c, m=m):
            e = M[c][m]["recovery"]
            lo, hi = e["ci95"]
            wide = (hi - lo) > 0.5          # ratio undetermined, not merely small
            v = 0.0 if abs(e["mean"]) < 5e-3 else e["mean"]
            return R + ("undet." if wide else f"${v:.2f}$") + "}"
        add(R + f"$R$, {lab}" + "} & " + cells(rec) + " \\\\")

    add("\\midrule")
    add("\\multicolumn{6}{l}{\\textit{Paired $\\Delta$AUC vs source-only "
        "(95\\% CI half-width, $\\times 10^{-3}$)}} \\\\")
    for m, lab in METHODS[1:]:
        add(f"{lab:14s} & " + cells(lambda c, m=m: delta(c, m, "auc")) + " \\\\")

    add("\\midrule")
    add("\\multicolumn{6}{l}{" + R + "\\textit{Macro-F1 at threshold 0.5}}} \\\\")
    for m, lab in METHODS:
        add(R + lab + "} & " + cells(lambda c, m=m: R + val(c, m, "f1") + "}") + " \\\\")
    add("\\multicolumn{6}{l}{" + R + "\\textit{Paired $\\Delta$macro-F1 vs source-only "
        "(95\\% CI half-width, $\\times 10^{-3}$)}}} \\\\")
    for m, lab in METHODS[1:]:
        add(R + lab + "} & " + cells(lambda c, m=m: R + delta(c, m, "f1") + "}") + " \\\\")

    add("\\midrule")
    add("\\multicolumn{6}{l}{\\textit{Q-factor at threshold 0.5}} \\\\")
    for m, lab in METHODS[:3]:
        add(f"{lab:14s} & " + cells(lambda c, m=m: val(c, m, "q05")) + " \\\\")

    add("\\bottomrule")
    add("\\end{tabular}}")
    add("\\end{adjustwidth}")
    add("\\end{table}")

    out = HERE / "manuscript" / "tables" / "tab1_full_results.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", out, f"({len(L)} lines)")

    # ---- companion table: the bootstrap intervals behind the R rows above ----
    K = []
    put = K.append

    def rci(c, m):
        e = M[c][m]["recovery"]
        lo, hi = e["ci95"]
        return (f"${zero(e['mean']):.2f}$ "
                f"$[{zero(lo):.2f}, {zero(hi):.2f}]$")

    put("\\begin{table}[H]")
    put("\\centering")
    put("\\small")
    put("\\caption{\\rev{Recovered fraction $R$ of the shift damage, with its 95\\% bootstrap "
        "interval (ratio of means, percentile bootstrap, $B = 10^4$ resamples of the 15 "
        "carrier--seed pairs; Section~\\ref{sec:scenario}). These are the intervals behind the "
        "$R$ rows of Table~\\ref{tab:full_results}. The MLP intervals span zero and are more "
        "than unit width, which is why that carrier is marked undetermined there: its damage is "
        "too small to divide by, not its recovery particularly "
        "uncertain.}}\\label{tab:recovery_ci}")
    put("\\begin{tabular}{lccc}")
    put("\\toprule")
    put(R + "Carrier} & " + R + "Mean matching} & " + R + "CORAL} & "
        + R + "MMD-rbf} \\\\")
    put("\\midrule")
    for c, lab in ORDER:
        put(R + lab + "} & "
            + " & ".join(R + rci(c, m) + "}" for m, _ in METHODS[1:]) + " \\\\")
    put("\\bottomrule")
    put("\\end{tabular}")
    put("\\end{table}")

    out2 = HERE / "manuscript" / "tables" / "tab_recovery_ci.tex"
    out2.write_text("\n".join(K) + "\n", encoding="utf-8")
    print("written", out2, f"({len(K)} lines)")


if __name__ == "__main__":
    main()
