"""
Single entry point: regenerate every table and figure from the raw per-seed measurements.

Runs the ablation (resuming from results/runs_<tag>.csv when it exists, so a clone with committed results needs
about a minute), then the power analysis for the same tag, then lists every figure written. Retrain everything with
ABLATE_FRESH=1; run the tiny smoke test with ABLATE_QUICK=1 (numbers meaningless).

Usage (repo root):  python scripts/reproduce.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env_setup import setup


def run(script, *args):
    print(f"[reproduce] {script} {' '.join(args)}".rstrip())
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    tag = setup(install=False)["tag"] + ("_quick" if os.environ.get("ABLATE_QUICK") == "1" else "")
    run("run_ablation.py")
    run("power_analysis.py", "--tag", tag)
    figures = sorted((ROOT / "results").glob(f"*_{tag}.png"))
    print(f"[reproduce] tag '{tag}': {len(figures)} figure(s) in results/")
    for fig in figures:
        print("  ", fig.relative_to(ROOT).as_posix())
