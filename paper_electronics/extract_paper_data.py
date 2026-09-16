"""Extract consolidated results for the Electronics paper figures.

Reads S0 manifest files and shift diagnostics, aggregates into a single
JSON file at paper_electronics/data/extracted_results.json.

Deltas versus source-only are computed as *paired* differences per seed:
every cell of the S0 matrix was run with the same seed set (42..56), and
the seed controls the train/test split and the classifier initialisation,
so the difference "method - srconly" is well defined seed by seed. We
report the paired mean, the paired standard error, a 95% t-interval
(df = n - 1), the paired t-test p-value and Cohen's d_z.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

CARRIERS = ["lr", "lgbm", "rf", "et", "mlp"]
METHODS = ["coral", "mean_matching", "mmd_linear", "mmd_rbf"]
METRICS = {"auc": "auc", "q": "q_factor", "f1": "f1_macro"}


def _carrier_from_experiment(exp: str) -> str:
    for tag in ("lgbm", "rf", "et", "mlp"):
        if exp.endswith(f"_{tag}"):
            return tag
    return "lr"


def _collect_s0_results(results_dir: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Collect S0 manifests as {cell_key: {seed: metrics}}.

    Manifests are read from the Hydra run directories and from the MLflow
    artifact store (``results/_mlruns``; the MMD cells only exist there).
    Each (cell, seed) pair is counted **once**: when the same seed was run
    more than once, the run with the latest ``timestamp_utc`` wins. This
    matters for the logistic-regression cells, whose first batch on
    2026-05-05 06:05 UTC used a superseded configuration.
    """
    latest: dict[tuple[str, int], str] = {}
    cells: dict[str, dict[int, dict[str, float]]] = {}
    # Shipped per-seed manifests (paper_electronics/data/manifests) are read
    # too, so the extraction works from a clean checkout without results/.
    shipped = results_dir.parent / "paper_electronics" / "data" / "manifests"
    paths = list(results_dir.rglob("manifest.json")) + list(shipped.rglob("seed*.json"))
    for mp in paths:
        with open(mp) as f:
            m = json.load(f)
        if m.get("scenario") != "S0":
            continue
        method = m["method"].replace("logreg-", "")
        clf = _carrier_from_experiment(m["experiment_name"])
        key = f"{method}__{clf}"
        seed = int(m["seed"])
        ts = str(m.get("timestamp_utc", ""))
        if ts < latest.get((key, seed), ""):
            continue
        latest[(key, seed)] = ts
        cells.setdefault(key, {})[seed] = m["metrics"]
    return cells


def _summarise(cells: dict[str, dict[int, dict[str, float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, by_seed in sorted(cells.items()):
        method, clf = key.split("__")
        entry: dict[str, Any] = {
            "method": method,
            "carrier": clf,
            "n_seeds": len(by_seed),
            "seeds": sorted(by_seed),
        }
        for short, name in METRICS.items():
            vals = np.array([by_seed[s][name] for s in sorted(by_seed)])
            entry[f"{short}_mean"] = round(float(vals.mean()), 4)
            entry[f"{short}_std"] = round(float(vals.std(ddof=1)), 4)
        summary[key] = entry
    return summary


def _paired_stats(diff: np.ndarray) -> dict[str, Any]:
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    if se > 0:
        p_val = stats.ttest_1samp(diff, 0.0).pvalue
        p_val = float(p_val)
        dz = mean / sd
    else:  # identical values on every seed
        p_val = 0.0 if mean != 0 else 1.0
        dz = float("inf") if mean != 0 else 0.0
    return {
        "mean": round(mean, 4),
        "se": round(float(se), 5),
        "ci95": [round(mean - tcrit * se, 4), round(mean + tcrit * se, 4)],
        "p_paired_t": p_val,
        "cohen_dz": round(float(dz), 2) if np.isfinite(dz) else None,
        "n_pairs": n,
    }


def _compute_deltas(cells: dict[str, dict[int, dict[str, float]]]) -> dict[str, Any]:
    """Paired deltas (method - srconly) per seed, for each carrier x method."""
    deltas: dict[str, Any] = {}
    for clf in CARRIERS:
        base = cells.get(f"srconly__{clf}")
        if not base:
            continue
        for method in METHODS:
            cell = cells.get(f"{method}__{clf}")
            if not cell:
                continue
            seeds = sorted(set(base) & set(cell))
            if not seeds:
                continue
            out: dict[str, Any] = {"method": method, "carrier": clf, "seeds": seeds}
            for short, name in METRICS.items():
                diff = np.array([cell[s][name] - base[s][name] for s in seeds])
                st = _paired_stats(diff)
                out[f"delta_{short}"] = st["mean"]
                out[f"delta_{short}_se"] = st["se"]
                out[f"delta_{short}_ci95"] = st["ci95"]
                out[f"delta_{short}_p"] = st["p_paired_t"]
                out[f"delta_{short}_dz"] = st["cohen_dz"]
                out[f"delta_{short}_per_seed"] = [round(float(d), 4) for d in diff]
            out["n_pairs"] = len(seeds)
            deltas[f"{method}__{clf}"] = out
    return deltas


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    output_path = project_root / "paper_electronics" / "data" / "extracted_results.json"

    cells = _collect_s0_results(results_dir)
    summary = _summarise(cells)
    deltas = _compute_deltas(cells)

    # Shift diagnostics (from session 8, hardcoded reference values)
    shift_diagnostics = {
        "mmd": {"value": 0.001114, "p_value": 0.005, "n_permutations": 200, "bandwidth": 13.9494},
        "proxy_a_distance": {"value": 0.2376, "cv_error": 0.4406},
        "calibration_drift": {"ece_source": 0.0171, "ece_target": 0.0189, "delta": 0.0018},
    }

    extracted = {
        "s0_summary": summary,
        "s0_deltas": deltas,
        "s0_shift_diagnostics": shift_diagnostics,
    }
    output_path.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    print(f"Extracted {len(summary)} cells, {len(deltas)} deltas -> {output_path}")
    for key in ("coral__lr", "coral__mlp", "coral__lgbm", "coral__et", "coral__rf"):
        d = deltas[key]
        print(
            f"{key:14s} dAUC={d['delta_auc']:+.4f} CI95={d['delta_auc_ci95']} "
            f"p={d['delta_auc_p']:.1e} dz={d['delta_auc_dz']}  n={d['n_pairs']}"
        )


if __name__ == "__main__":
    main()
