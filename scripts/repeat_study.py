"""
Repeat the whole study from scratch N times on this machine and report how much the headline numbers move.

Each repeat retrains all (arm, seed) runs and re-times latency (ABLATE_FRESH=1), then archives that repeat's
outputs under results/repeats/<tag>/rep<k>/. The summary answers two separate questions:
  1. Determinism: are per-run test MAEs bit-identical between repeats? (the README promises this on one machine)
  2. Stability:   how much do arm means, latency and training time move between repeats?

Usage (repo root):
    python scripts/repeat_study.py --n 3            # about 3 x 8 min on a Colab T4
    ABLATE_QUICK=1 python scripts/repeat_study.py --n 3   # smoke test of this harness only (numbers meaningless)

Finished repeats are skipped on a rerun, so a dropped Colab session resumes where it stopped.
Writes results/repeats/<tag>/summary.csv and summary.json. It never touches the committed single-run results:
the entry point is run in a scratch copy of results/ that is restored afterwards.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
QUICK = os.environ.get("ABLATE_QUICK") == "1"


def run_once(k: int, backup: Path) -> str:
    """Run the entry point fresh and return the tag it wrote."""
    before = {p.name for p in RESULTS.glob("runs_*.csv")}
    env = dict(os.environ, ABLATE_FRESH="1")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_ablation.py")], cwd=ROOT, env=env, check=True)
    after = sorted(RESULTS.glob("runs_*.csv"), key=lambda p: p.stat().st_mtime)
    tag = after[-1].stem.removeprefix("runs_")
    dest = RESULTS / "repeats" / tag / f"rep{k}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in RESULTS.glob(f"*_{tag}.*"):
        if p.suffix in {".csv", ".json"}:
            shutil.copy2(p, dest / p.name)
    # restore whatever was committed under this tag so repeats never overwrite the single-run record
    for p in backup.glob(f"*_{tag}.*"):
        shutil.copy2(p, RESULTS / p.name)
    for name in {p.name for p in RESULTS.glob(f"*_{tag}.*")} - {p.name for p in backup.glob(f"*_{tag}.*")}:
        (RESULTS / name).unlink()
    print(f"[repeat] rep{k} archived to {dest.relative_to(ROOT)}")
    return tag


def summarise(tag: str, n: int) -> None:
    base = RESULTS / "repeats" / tag
    runs = [pd.read_csv(base / f"rep{k}" / f"runs_{tag}.csv").sort_values(["arm", "seed"]).reset_index(drop=True) for k in range(n)]
    lat = [pd.read_csv(base / f"rep{k}" / f"summary_{tag}.csv").set_index("arm") for k in range(n)]
    mae = np.stack([r["test_mae_kw"].to_numpy() for r in runs])
    max_abs = float(np.abs(mae - mae[0]).max())
    rows = []
    for arm in runs[0]["arm"].unique():
        per_rep_mean = [r[r["arm"] == arm]["test_mae_kw"].mean() for r in runs]
        rows.append({
            "arm": arm, "repeats": n,
            "mae_mean_of_means": np.mean(per_rep_mean), "mae_mean_range_across_repeats": max(per_rep_mean) - min(per_rep_mean),
            "p50_ms_mean": np.mean([s.loc[arm, "p50_ms_window"] for s in lat]),
            "p50_ms_range": max(s.loc[arm, "p50_ms_window"] for s in lat) - min(s.loc[arm, "p50_ms_window"] for s in lat),
            "train_s_mean": np.mean([r[r["arm"] == arm]["train_s"].mean() for r in runs]),
            "train_s_range": max(r[r["arm"] == arm]["train_s"].mean() for r in runs) - min(r[r["arm"] == arm]["train_s"].mean() for r in runs),
        })
    out = pd.DataFrame(rows)
    out.to_csv(base / "summary.csv", index=False)
    (base / "summary.json").write_text(json.dumps({
        "tag": tag, "repeats": n, "quick": QUICK, "max_abs_diff_per_run_test_mae_kw": max_abs,
        "bit_identical_across_repeats": bool(max_abs == 0.0)}, indent=1), encoding="utf-8")
    print(out.round(4).to_string(index=False))
    print(f"\nmax |per-run test MAE difference| between repeats: {max_abs:.3e} kW "
          f"({'bit-identical' if max_abs == 0 else 'NOT bit-identical'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    n = ap.parse_args().n
    backup = RESULTS.parent / ".results_backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(RESULTS, backup, ignore=shutil.ignore_patterns("repeats"))
    try:
        sys.path.insert(0, str(ROOT))
        from env_setup import setup

        tag = setup(install=False)["tag"] + ("_quick" if QUICK else "")  # the tag the entry point will write
        for k in range(n):
            if (RESULTS / "repeats" / tag / f"rep{k}" / f"runs_{tag}.csv").exists():
                print(f"[repeat] rep{k} already archived, skipping (delete its folder to redo it)")
                continue
            run_once(k, backup)
        summarise(tag, n)
    finally:
        shutil.rmtree(backup, ignore_errors=True)
