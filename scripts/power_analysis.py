"""
Minimum detectable effect and seed budget, computed from the committed per-seed measurements.

Question answered: with the seed spread actually observed, how large a test-MAE difference could 5 seeds per arm
have detected, and how many seeds (and how many T4 seconds) would it take to resolve the differences that the
ablation could not separate from noise?

Usage (repo root):  python scripts/power_analysis.py [--tag t4gpu]
Reads   results/runs_<tag>.csv
Writes  results/power_<tag>.csv, results/power_<tag>.json, results/power_<tag>.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ALPHA, POWER, N_COMPARISONS = 0.05, 0.80, 3  # Holm's first step is alpha / 3, the strictest it applies here


def mde(sd: float, n: int, alpha: float) -> float:
    """Smallest true difference in means detected with probability POWER by a two-sided two-sample t-test."""
    df = 2 * n - 2
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    scale = sd * np.sqrt(2 / n)
    lo, hi = 0.0, 50 * sd
    for _ in range(80):  # bisection on the noncentral-t power
        mid = (lo + hi) / 2
        power = 1 - stats.nct.cdf(tcrit, df, mid / scale) + stats.nct.cdf(-tcrit, df, mid / scale)
        lo, hi = (mid, hi) if power < POWER else (lo, mid)
    return (lo + hi) / 2


def seeds_needed(sd: float, delta: float, alpha: float) -> int:
    for n in range(2, 5000):
        if mde(sd, n, alpha) <= delta:
            return n
    return -1


def main(tag: str) -> None:
    runs = pd.read_csv(ROOT / "results" / f"runs_{tag}.csv")
    arms = [a for a in runs["arm"].unique() if a != "none"]  # 'none' is far outside the noise; it is not the open question
    sds = runs[runs["arm"].isin(arms)].groupby("arm")["test_mae_kw"].std(ddof=1)
    n_obs = int(runs.groupby("arm")["seed"].nunique().min())
    sd_pool = float(np.sqrt((sds ** 2).mean()))  # all arms have the same n, so the pooled variance is the mean variance
    train_s = float(runs[runs["arm"].isin(arms)]["train_s"].mean())

    rows = []
    for label, alpha in (("uncorrected a=0.05", ALPHA), ("Holm-strict a=0.05/3", ALPHA / N_COMPARISONS)):
        rows.append({"alpha": label, "n_seeds": n_obs, "pooled_sd_kw": sd_pool, "mde_kw": mde(sd_pool, n_obs, alpha),
                     "mde_pct_of_reference": 100 * mde(sd_pool, n_obs, alpha) / runs[runs["arm"] == "sinusoidal"]["test_mae_kw"].mean()})
    for delta in (0.25, 0.15, 0.10, 0.05):
        n = seeds_needed(sd_pool, delta, ALPHA / N_COMPARISONS)
        rows.append({"alpha": "Holm-strict a=0.05/3", "target_delta_kw": delta, "n_seeds": n, "pooled_sd_kw": sd_pool,
                     "t4_seconds_4_arms": n * 4 * train_s, "t4_minutes_4_arms": n * 4 * train_s / 60})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / f"power_{tag}.csv", index=False)

    obs = {}
    ref = runs[runs["arm"] == "sinusoidal"]["test_mae_kw"]
    for a in arms:
        if a != "sinusoidal":
            obs[a] = abs(runs[runs["arm"] == a]["test_mae_kw"].mean() - ref.mean())
    (ROOT / "results" / f"power_{tag}.json").write_text(json.dumps({
        "tag": tag, "alpha": ALPHA, "power": POWER, "pooled_sd_kw": sd_pool, "per_arm_sd_kw": sds.to_dict(),
        "n_seeds_observed": n_obs, "mde_kw_uncorrected": mde(sd_pool, n_obs, ALPHA),
        "mde_kw_holm_strict": mde(sd_pool, n_obs, ALPHA / N_COMPARISONS), "observed_abs_diff_kw": obs,
        "train_s_mean_nonbaseline": train_s}, indent=1), encoding="utf-8")

    ns = np.arange(3, 61)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax.plot(ns, [mde(sd_pool, n, ALPHA / N_COMPARISONS) for n in ns], color="#1b6ca8", lw=2, label="Holm-strict (α = 0.05/3)")
    ax.plot(ns, [mde(sd_pool, n, ALPHA) for n in ns], color="#1b6ca8", lw=1.2, ls="--", label="uncorrected (α = 0.05)")
    for a, d in obs.items():
        ax.axhline(d, color="#b5651d", lw=1)
        ax.text(6.5, d - 0.018, f"observed |{a} − sinusoidal| = {d:.3f} kW", color="#b5651d", fontsize=8, ha="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.axvline(n_obs, color="#555", lw=0.8, ls=":")
    ax.text(n_obs + 0.4, ax.get_ylim()[1] * 0.92, f"this study: {n_obs} seeds/arm", fontsize=8, color="#555")
    ax.set_xlabel("seeds per arm")
    ax.set_ylabel(f"minimum detectable difference in test MAE (kW)\n80% power, pooled seed SD = {sd_pool:.3f} kW")
    ax.set_title("What the seed budget could and could not resolve", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"power_{tag}.png")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t4gpu")
    main(ap.parse_args().tag)
