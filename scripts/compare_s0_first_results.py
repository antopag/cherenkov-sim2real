"""Compare srconly vs CORAL results on S0.

Reads manifest.json files from the Hydra run directories under
results/s0_srconly/ and results/s0_coral/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _collect_manifests(output_dir: Path) -> list[dict]:  # type: ignore[type-arg]
    """Collect all manifest.json files from Hydra run dirs."""
    results = []
    for f in output_dir.rglob("manifest.json"):
        with open(f, encoding="utf-8") as fp:
            results.append(json.load(fp))
    return sorted(results, key=lambda r: r["seed"])


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    srconly_results = _collect_manifests(project_root / "results" / "s0_srconly")
    coral_results = _collect_manifests(project_root / "results" / "s0_coral")

    print("=" * 70)
    print("S0 FIRST RESULTS: srconly vs CORAL (logistic regression)")
    print("=" * 70)
    print()

    # Detailed table
    print(f"{'Method':<12} {'Seed':<6} {'F1':<8} {'AUC':<8} {'Q-factor':<10}")
    print("-" * 44)
    for r in srconly_results:
        m = r["metrics"]
        print(f"{'srconly':<12} {r['seed']:<6} {m['f1_macro']:<8.4f} {m['auc']:<8.4f} {m['q_factor']:<10.4f}")
    print("-" * 44)
    for r in coral_results:
        m = r["metrics"]
        print(f"{'coral':<12} {r['seed']:<6} {m['f1_macro']:<8.4f} {m['auc']:<8.4f} {m['q_factor']:<10.4f}")
    print("-" * 44)
    print()

    # Summary
    def _summarize(results: list[dict]) -> dict[str, tuple[float, float]]:  # type: ignore[type-arg]
        metrics_arrays: dict[str, list[float]] = {"f1_macro": [], "auc": [], "q_factor": []}
        for r in results:
            for k in metrics_arrays:
                metrics_arrays[k].append(r["metrics"][k])
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics_arrays.items()}

    srconly_summary = _summarize(srconly_results)
    coral_summary = _summarize(coral_results)

    print(f"SUMMARY (mean +/- std, {len(srconly_results)} seeds)")
    print(f"{'Method':<12} {'F1':<20} {'AUC':<20} {'Q-factor':<20}")
    print("-" * 72)
    for name, s in [("srconly", srconly_summary), ("coral", coral_summary)]:
        f1 = f"{s['f1_macro'][0]:.4f} +/- {s['f1_macro'][1]:.4f}"
        auc = f"{s['auc'][0]:.4f} +/- {s['auc'][1]:.4f}"
        q = f"{s['q_factor'][0]:.4f} +/- {s['q_factor'][1]:.4f}"
        print(f"{name:<12} {f1:<20} {auc:<20} {q:<20}")
    print("-" * 72)

    # Delta
    print()
    for metric in ["f1_macro", "auc", "q_factor"]:
        delta = coral_summary[metric][0] - srconly_summary[metric][0]
        direction = "+" if delta > 0 else ""
        print(f"  CORAL - srconly ({metric}): {direction}{delta:.4f}")


if __name__ == "__main__":
    main()
