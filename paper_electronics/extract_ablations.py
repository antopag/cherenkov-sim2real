"""Extract the revision-round ablations (scripts/run_ablations.py).

Reads ``results/abl_*/**/manifest.json`` and writes
``paper_electronics/data/ablations.json`` with, for every ablation cell,
the per-seed metrics and the *paired* delta versus the matching
source-only cell (same config, same carrier, same seed), using the same
statistics as ``extract_paper_data.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

METRICS = {"auc": "auc", "q": "q_factor"}


def _paired(diff: np.ndarray) -> dict[str, Any]:
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else float("nan")
    p = float(stats.ttest_1samp(diff, 0.0).pvalue) if se > 0 else (0.0 if mean else 1.0)
    return {
        "mean": round(mean, 4),
        "se": round(float(se), 5),
        "ci95": [round(mean - tcrit * se, 4), round(mean + tcrit * se, 4)],
        "p": p,
        "n": n,
    }


def _collect(results_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """{experiment_name: {seed: manifest}} keeping the latest run per seed."""
    cells: dict[str, dict[int, dict[str, Any]]] = {}
    shipped = results_dir.parent / "paper_electronics" / "data" / "manifests"
    for exp_dir in sorted(results_dir.glob("abl_*")) + sorted(shipped.glob("abl_*")):
        for mp in list(exp_dir.glob("*/manifest.json")) + list(exp_dir.glob("seed*.json")):
            m = json.loads(mp.read_text(encoding="utf-8"))
            seed = int(m["seed"])
            prev = cells.setdefault(exp_dir.name, {}).get(seed)
            if prev is None or m["timestamp_utc"] > prev["timestamp_utc"]:
                cells[exp_dir.name][seed] = m
    return cells


def _summary(by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(by_seed), "seeds": sorted(by_seed)}
    for short, name in METRICS.items():
        v = np.array([by_seed[s]["metrics"][name] for s in sorted(by_seed)])
        out[f"{short}_mean"] = round(float(v.mean()), 4)
        out[f"{short}_std"] = round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0
    return out


def _delta(cell: dict[int, dict[str, Any]], base: dict[int, dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted(set(cell) & set(base))
    out: dict[str, Any] = {"seeds": seeds}
    for short, name in METRICS.items():
        diff = np.array([cell[s]["metrics"][name] - base[s]["metrics"][name] for s in seeds])
        st = _paired(diff)
        out[f"delta_{short}"] = st["mean"]
        out[f"delta_{short}_ci95"] = st["ci95"]
        out[f"delta_{short}_p"] = st["p"]
        out[f"delta_{short}_per_seed"] = [round(float(d), 4) for d in diff]
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cells = _collect(root / "results")

    # --- A. depth sweep (RF) -------------------------------------------------
    depth: dict[str, Any] = {}
    for exp, by_seed in cells.items():
        if not exp.startswith("abl_depth_"):
            continue
        tag = exp.split("_")[2]  # 02 .. 16 or none
        method = "_".join(exp.split("_")[3:-1])
        depth.setdefault(tag, {})[method] = _summary(by_seed)
    for tag, methods in depth.items():
        base = cells[f"abl_depth_{tag}_srconly_rf"]
        for method in ("mean_matching", "coral"):
            methods[method].update(_delta(cells[f"abl_depth_{tag}_{method}_rf"], base))
        # CORAL minus mean matching, paired
        methods["coral_minus_mm"] = _delta(
            cells[f"abl_depth_{tag}_coral_rf"], cells[f"abl_depth_{tag}_mean_matching_rf"]
        )
        methods["max_depth"] = None if tag == "none" else int(tag)

    # --- B. shift type / intensity (all carriers) ---------------------------
    shift: dict[str, Any] = {}
    for exp, by_seed in cells.items():
        if not exp.startswith("abl_shift_"):
            continue
        parts = exp.split("_")  # abl shift <cfg> <method...> <carrier>
        cfg, carrier = parts[2], parts[-1]
        method = "_".join(parts[3:-1])
        shift.setdefault(cfg, {}).setdefault(carrier, {})[method] = _summary(by_seed)
    for cfg, carriers in shift.items():
        for carrier, methods in carriers.items():
            base = cells[f"abl_shift_{cfg}_srconly_{carrier}"]
            for method in ("mean_matching", "coral"):
                methods[method].update(_delta(cells[f"abl_shift_{cfg}_{method}_{carrier}"], base))

    out = root / "paper_electronics" / "data" / "ablations.json"
    out.write_text(json.dumps({"depth": depth, "shift": shift}, indent=2), encoding="utf-8")
    print(f"depth tags: {sorted(depth)}  shift configs: {sorted(shift)} -> {out}")
    for tag in sorted(depth, key=lambda t: (t == "none", t)):
        d = depth[tag]
        print(
            f"depth={d['max_depth']!s:>4s}  srconly AUC={d['srconly']['auc_mean']:.4f}  "
            f"dAUC mm={d['mean_matching']['delta_auc']:+.4f}  coral={d['coral']['delta_auc']:+.4f}  "
            f"coral-mm={d['coral_minus_mm']['delta_auc']:+.4f} {d['coral_minus_mm']['delta_auc_ci95']}"
        )
    for cfg in sorted(shift):
        for carrier in ("lr", "mlp", "lgbm", "et", "rf"):
            if carrier in shift[cfg]:
                c = shift[cfg][carrier]
                print(
                    f"shift={cfg:7s} {carrier:4s} srconly AUC={c['srconly']['auc_mean']:.4f}  "
                    f"dAUC mm={c['mean_matching']['delta_auc']:+.4f}  coral={c['coral']['delta_auc']:+.4f} "
                    f"{c['coral']['delta_auc_ci95']}"
                )


if __name__ == "__main__":
    main()
