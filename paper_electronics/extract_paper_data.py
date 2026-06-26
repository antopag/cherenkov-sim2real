"""Extract consolidated results for the Electronics paper figures.

Reads S0 manifest files and shift diagnostics, aggregates into a single
JSON file at paper_electronics/data/extracted_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _collect_s0_results(results_dir: Path) -> dict[str, Any]:
    """Collect all S0 manifests, grouped by (method, carrier)."""
    cells: dict[str, list[dict[str, float]]] = {}

    for mp in results_dir.rglob("manifest.json"):
        with open(mp) as f:
            m = json.load(f)
        if m.get("scenario") != "S0":
            continue
        method = m["method"].replace("logreg-", "")
        exp = m["experiment_name"]
        if "_lgbm" in exp:
            clf = "lgbm"
        elif "_rf" in exp:
            clf = "rf"
        elif "_et" in exp:
            clf = "et"
        elif "_mlp" in exp:
            clf = "mlp"
        else:
            clf = "lr"

        key = f"{method}__{clf}"
        cells.setdefault(key, []).append(m["metrics"])

    # Aggregate
    summary: dict[str, Any] = {}
    for key, metrics_list in sorted(cells.items()):
        method, clf = key.split("__")
        f1s = [m["f1_macro"] for m in metrics_list]
        aucs = [m["auc"] for m in metrics_list]
        qs = [m["q_factor"] for m in metrics_list]
        summary[key] = {
            "method": method,
            "carrier": clf,
            "n_seeds": len(metrics_list),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std": round(float(np.std(aucs)), 4),
            "q_mean": round(float(np.mean(qs)), 4),
            "q_std": round(float(np.std(qs)), 4),
        }

    return summary


def _compute_deltas(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Compute deltas vs srconly for each carrier x DA method."""
    deltas: dict[str, Any] = {}
    carriers = ["lr", "lgbm", "rf", "et", "mlp"]
    methods = ["coral", "mean_matching", "mmd_linear", "mmd_rbf"]

    for clf in carriers:
        base_key = f"srconly__{clf}"
        if base_key not in summary:
            continue
        base = summary[base_key]
        for method in methods:
            key = f"{method}__{clf}"
            if key not in summary:
                continue
            cell = summary[key]
            n = min(cell["n_seeds"], base["n_seeds"])
            dauc = cell["auc_mean"] - base["auc_mean"]
            dq = cell["q_mean"] - base["q_mean"]
            # Approximate sigma of the difference
            se_auc = np.sqrt(cell["auc_std"] ** 2 / n + base["auc_std"] ** 2 / n)
            se_q = np.sqrt(cell["q_std"] ** 2 / n + base["q_std"] ** 2 / n)
            sig_auc = float(dauc / se_auc) if se_auc > 0 else 0.0
            sig_q = float(dq / se_q) if se_q > 0 else 0.0

            deltas[f"{method}__{clf}"] = {
                "method": method,
                "carrier": clf,
                "delta_auc": round(dauc, 4),
                "delta_auc_sigma": round(sig_auc, 1),
                "delta_q": round(dq, 4),
                "delta_q_sigma": round(sig_q, 1),
            }

    return deltas


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    output_path = project_root / "paper_electronics" / "data" / "extracted_results.json"

    summary = _collect_s0_results(results_dir)
    deltas = _compute_deltas(summary)

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


if __name__ == "__main__":
    main()
