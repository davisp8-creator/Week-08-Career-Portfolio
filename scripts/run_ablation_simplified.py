# Step 1: Environment Setup
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))
from env_setup import setup, hardware_snapshot

ENV = setup(require_t4=False)  # set require_t4=True to refuse anything but a Colab T4
if not (ROOT / ".git").exists():
    pass


# Step 2: Imports, Single-Source Constants & Hardware-Tagged Output Paths
# Install required libraries: run the installer for your OS (or `python env_setup.py`);
# on Colab, Step 1 installs anything missing and leaves the CUDA torch build alone.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before CUDA initialises

import hashlib
import json
import math
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import stats

T_START = time.perf_counter()
IN_NOTEBOOK = "ipykernel" in sys.modules
QUICK = os.environ.get("ABLATE_QUICK") == "1"  # smoke test only: results NOT meaningful
FRESH = os.environ.get("ABLATE_FRESH") == "1"  # ignore saved runs and retrain everything


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


if not (ROOT / "configs" / "ablation.yaml").exists():
    raise FileNotFoundError(f"{ROOT / 'configs' / 'ablation.yaml'} not found. The working directory must be the repo root "
                            "(run from a clone of this repository).")
CFG = yaml.safe_load((ROOT / "configs" / "ablation.yaml").read_text(encoding="utf-8-sig"))
CFG = deep_merge({k: v for k, v in CFG.items() if k != "quick"}, CFG["quick"]) if QUICK else \
    {k: v for k, v in CFG.items() if k != "quick"}
ARMS, SEEDS, REF = CFG["arms"], CFG["seeds"], CFG["reference_arm"]
DEVICE = torch.device(ENV["device"])
TAG = ENV["tag"] + ("_quick" if QUICK else "")
RESULTS, DATA_DIR = ROOT / "results", ROOT / "data"
RESULTS.mkdir(exist_ok=True)
torch.set_num_threads(CFG["threads"])
torch.use_deterministic_algorithms(True, warn_only=True)
print(f"arms={ARMS} seeds={SEEDS} tag={TAG} device={DEVICE} quick={QUICK}")


# Step 3: Synthetic Demand Data
def generate_demand(dc):
    rng = np.random.default_rng(dc["data_seed"])
    n_hours = dc["n_days"] * 24
    t = np.arange(n_hours)
    idx = pd.Timestamp(dc["start"]) + pd.to_timedelta(t, unit="h")
    hod, dow = idx.hour.values, idx.day_of_week.values
    annual = 12.0 - 11.0 * np.cos(2 * np.pi * (t - 400) / 8760)
    diurnal = 4.0 * np.sin(2 * np.pi * (t % 24 - 9) / 24)
    eps = rng.normal(0, 1.0, n_hours)
    ar = np.zeros(n_hours)
    for i in range(1, n_hours):  # AR(1) weather noise with day-scale persistence
        ar[i] = 0.97 * ar[i - 1] + eps[i]
    temp = annual + diurnal + 0.9 * ar
    holidays = set(rng.choice(dc["n_days"], size=dc["n_days"] // 40, replace=False).tolist())
    is_hol = np.array([(d // 24) in holidays for d in t])
    frames = []
    for f in range(dc["n_feeders"]):
        base = rng.uniform(80, 400)
        res_share = rng.uniform(0.3, 0.9)
        phase = rng.normal(0, 1.2)
        res = (0.6 + 0.5 * np.exp(-0.5 * ((hod - 7.5 - phase) / 1.6) ** 2)
               + 0.9 * np.exp(-0.5 * ((hod - 19 - phase) / 2.2) ** 2))
        com = 0.5 + 0.8 / (1 + np.exp(-(hod - 7.5))) - 0.8 / (1 + np.exp(-(hod - 17.5)))
        shape = res_share * res + (1 - res_share) * com
        weekly = np.where(dow >= 5, rng.uniform(0.82, 0.95), 1.0)
        weekly = np.where(is_hol, 0.85, weekly)
        ft = temp + rng.normal(0, 0.8)
        climate = (1 + rng.uniform(0.012, 0.03) * np.maximum(15 - ft, 0)
                   + rng.uniform(0.004, 0.015) * np.maximum(ft - 22, 0))
        irrigation = 1 + rng.uniform(0.0, 0.3) * ((idx.month.values >= 6) & (idx.month.values <= 8) & (hod >= 22) | (hod <= 4))
        y = base * shape * weekly * climate * irrigation * (1 + 0.04 * rng.normal(size=n_hours))
        for s in rng.choice(n_hours - 12, size=max(1, n_hours // 6000), replace=False):  # rare outages
            y[s:s + rng.integers(2, 8)] *= rng.uniform(0.0, 0.2)
        frames.append(pd.DataFrame({"feeder": f, "timestamp": idx, "load_kw": y.round(3), "temp_c": ft.round(2)}))
    return pd.concat(frames, ignore_index=True)


def build_dataset(dc, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("demand_quick.csv" if QUICK else "demand.csv")
    df = generate_demand(dc)
    df.to_csv(path, index=False, lineterminator="\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".meta.json").write_text(json.dumps({"sha256": sha, "rows": len(df), "config": dc}, indent=2))
    return df, sha


DF, DATA_SHA = build_dataset(CFG["data"], DATA_DIR)
print(f"data: {len(DF):,} rows, {CFG['data']['n_feeders']} feeders, sha256={DATA_SHA[:16]}...")


# Step 4: Windows and Chronological Splits
def make_windows(df, cfg):
    L, H = cfg["window"]["lookback"], cfg["window"]["horizon"]
    out = {k: ([], [], []) for k in ("train", "val", "test")}  # X, Y, load std in kW
    for _, g in df.groupby("feeder"):
        v = g.sort_values("timestamp")[["load_kw", "temp_c"]].to_numpy(np.float32)
        n = len(v)
        a, b = int(n * cfg["split"]["train"]), int(n * (cfg["split"]["train"] + cfg["split"]["val"]))
        mu, sd = v[:a].mean(0), v[:a].std(0) + 1e-6
        z = (v - mu) / sd
        for name, (lo, hi), stride in (("train", (0, a), cfg["window"]["stride_train"]),
                                       ("val", (a, b), cfg["window"]["stride_eval"]),
                                       ("test", (b, n), cfg["window"]["stride_eval"])):
            for s in range(lo, hi - L - H + 1, stride):
                out[name][0].append(z[s:s + L])
                out[name][1].append(z[s + L:s + L + H, 0])
                out[name][2].append(sd[0])
    return {k: tuple(torch.tensor(np.array(x), dtype=torch.float32) for x in v) for k, v in out.items()}


def naive_baselines(X, Y, S):
    day, week = X[:, -24:, 0], X[:, :24, 0]  # lookback is exactly 168 h, so index k is one week before target hour k
    return {"naive_day_mae_kw": ((day - Y).abs() * S[:, None]).mean().item(),
            "naive_week_mae_kw": ((week - Y).abs() * S[:, None]).mean().item()}


W = make_windows(DF, CFG)
NAIVE = naive_baselines(*W["test"])
print({k: tuple(v[0].shape) for k, v in W.items()}, {k: round(v, 2) for k, v in NAIVE.items()})


# Step 5: Forecaster with Four Positional Encodings, and Reference Checks
def sinusoidal_table(length, d):
    pos = torch.arange(length).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def rope_tables(length, head_dim):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.arange(length).float().unsqueeze(1) * inv.unsqueeze(0)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: batch, heads, length, head_dim
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = x1 * cos - x2 * sin, x1 * sin + x2 * cos
    return out


class Block(nn.Module):
    def __init__(self, d, heads, ff, dropout, rope):
        super().__init__()
        self.h, self.hd, self.rope = heads, d // heads, rope
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def attend(self, x, cos, sin):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.hd), dim=-1)
        return (att @ v).transpose(1, 2).reshape(B, L, D), (q, k, v)

    def forward(self, x, cos, sin):
        y, _ = self.attend(x, cos, sin)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.n2(x)))


class Forecaster(nn.Module):
    def __init__(self, pos_enc, n_in, lookback, horizon, d=64, heads=4, layers=2, ff=128, dropout=0.1):
        super().__init__()
        assert pos_enc in {"none", "sinusoidal", "learned", "rope"}
        self.pos_enc = pos_enc
        self.inp = nn.Linear(n_in, d)
        if pos_enc == "sinusoidal":
            self.register_buffer("pe", sinusoidal_table(lookback, d))
        elif pos_enc == "learned":
            self.pe = nn.Parameter(torch.randn(lookback, d) * 0.02)
        if pos_enc == "rope":
            c, s = rope_tables(lookback, d // heads)
            self.register_buffer("cos", c)
            self.register_buffer("sin", s)
        else:
            self.cos = self.sin = None
        self.blocks = nn.ModuleList(Block(d, heads, ff, dropout, pos_enc == "rope") for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, horizon)

    def forward(self, x):
        h = self.inp(x)
        if self.pos_enc in ("sinusoidal", "learned"):
            h = h + self.pe
        for b in self.blocks:
            h = b(h, self.cos, self.sin)
        return self.head(self.norm(h)[:, -1])  # last-token readout is position-sensitive


def build_model(arm):
    return Forecaster(arm, n_in=2, lookback=CFG["window"]["lookback"], horizon=CFG["window"]["horizon"], **CFG["model"])


def reference_checks():
    torch.manual_seed(0)
    blk = Block(32, 4, 64, 0.0, rope=False).eval()
    x = torch.randn(3, 20, 32)
    mine, (q, k, v) = blk.attend(x, None, None)
    ref = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(3, 20, 32)
    attn_diff = (mine - ref).abs().max().item()
    c, s = rope_tables(40, 8)
    q0, k0 = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def score(i, j):
        return (apply_rope(q0, c[i:i + 1], s[i:i + 1]) * apply_rope(k0, c[j:j + 1], s[j:j + 1])).sum().item()

    rope_diff = abs(score(5, 2) - score(25, 22))  # same offset (3), different absolute position
    print(f"reference check: hand-built attention vs F.scaled_dot_product_attention max|diff| = {attn_diff:.2e}")
    print(f"reference check: RoPE score at offset 3, positions (5,2) vs (25,22): |diff| = {rope_diff:.2e}")
    assert attn_diff < 1e-5 and rope_diff < 1e-5, "reference check failed"
    return {"attention_max_abs_diff": attn_diff, "rope_offset_invariance_diff": rope_diff}


CHECKS = reference_checks()
print({a: sum(p.numel() for p in build_model(a).parameters()) for a in ARMS})


# Step 6: Train and Evaluate One (Arm, Seed)
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, X, Y, S, bs=512):
    model.eval()
    err = torch.cat([(model(X[i:i + bs].to(DEVICE)).cpu() - Y[i:i + bs]).abs() * S[i:i + bs, None]
                     for i in range(0, len(X), bs)])
    return {"mae_kw": err.mean().item(), "rmse_kw": err.pow(2).mean().sqrt().item()}


def run_one(arm, seed):
    set_seed(seed)
    t = CFG["train"]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        floor = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    model = build_model(arm).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    Xtr, Ytr, _ = W["train"]
    steps = t["epochs"] * math.ceil(len(Xtr) / t["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)
    rss0 = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    best, best_state, curve = 1e9, None, []
    for _ in range(t["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(perm), t["batch_size"]):
            idx = perm[i:i + t["batch_size"]]
            loss = F.l1_loss(model(Xtr[idx].to(DEVICE)), Ytr[idx].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            sched.step()
        v = evaluate(model, *W["val"])["mae_kw"]
        curve.append(v)
        if v < best:  # same checkpoint rule for every arm
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    te = evaluate(model, *W["test"])
    peak = (torch.cuda.max_memory_allocated() - floor) / 2**20 if DEVICE.type == "cuda" else float("nan")
    return {"arm": arm, "seed": seed, "n_params": n_params, "val_mae_kw": best, "test_mae_kw": te["mae_kw"],
            "test_rmse_kw": te["rmse_kw"], "best_epoch": int(np.argmin(curve)) + 1, "train_s": train_s,
            "peak_gpu_mb": peak, "rss_delta_mb": (psutil.Process().memory_info().rss - rss0) / 2**20}


# Step 7: The Ablation Loop
RUNS_CSV = RESULTS / f"runs_{TAG}.csv"
rows = [] if FRESH or not RUNS_CSV.exists() else pd.read_csv(RUNS_CSV).to_dict("records")
done = {(r["arm"], int(r["seed"])) for r in rows}
for seed in SEEDS:
    order = list(np.random.default_rng(1000 + seed).permutation(ARMS))
    for arm in order:
        if (arm, seed) in done:
            continue
        r = run_one(arm, seed)
        rows.append(r)
        pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)
        print(f"seed={seed} {arm:11s} test_mae={r['test_mae_kw']:.3f} kW  train={r['train_s']:.0f}s")
RUNS = pd.DataFrame(rows)


# Step 8: Latency Pass (Interleaved)
def time_forward(model, xb):
    lc = CFG["latency"]
    out = []
    with torch.no_grad():
        for r in range(lc["warmup"] + lc["repeats"]):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            s = time.perf_counter()
            model(xb)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            if r >= lc["warmup"]:
                out.append((time.perf_counter() - s) * 1000)
    return np.percentile(out, 50), np.percentile(out, 95)


LAT_CSV = RESULTS / f"latency_runs_{TAG}.csv"
if LAT_CSV.exists() and not FRESH:  # a resume reuses the saved measurement, like it reuses saved training runs
    LAT = pd.read_csv(LAT_CSV)
    print(f"latency: reusing {LAT_CSV.name} (ABLATE_FRESH=1 measures again)")
else:
    set_seed(0)
    lat_models = {a: build_model(a).to(DEVICE).eval() for a in ARMS}
    lat_x = W["test"][0][:CFG["latency"]["batch_size"]].to(DEVICE)
    lat_rows = []
    rng_lat = np.random.default_rng(2024)
    for rnd in range(CFG["latency"]["rounds"]):
        for arm in rng_lat.permutation(ARMS):
            p50, p95 = time_forward(lat_models[arm], lat_x)
            lat_rows.append({"arm": arm, "round": rnd, "p50_ms_window": p50 / len(lat_x), "p95_ms_window": p95 / len(lat_x)})
    LAT = pd.DataFrame(lat_rows)
    LAT.to_csv(LAT_CSV, index=False)
print(LAT.groupby("arm")[["p50_ms_window", "p95_ms_window"]].mean().reindex(ARMS).round(4))


# Step 9: Statistics and Verdict
def ci95(x):
    x = np.asarray(x, float)
    return float("nan") if len(x) < 2 else stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def holm(p):
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


metric = "test_mae_kw"
g = RUNS.groupby("arm")
lat_mean = LAT.groupby("arm")[["p50_ms_window", "p95_ms_window"]].mean()
lat_spread = LAT.groupby("arm")["p50_ms_window"].std(ddof=1)
SUMM = pd.DataFrame({
    "n_seeds": g.size(), "n_params": g.n_params.first(),
    "mae_mean": g[metric].mean(), "mae_sd": g[metric].std(ddof=1), "mae_ci95_halfwidth": g[metric].apply(ci95),
    "mae_min": g[metric].min(), "mae_max": g[metric].max(), "rmse_mean": g.test_rmse_kw.mean(),
    "train_s_mean": g.train_s.mean(), "p50_ms_window": lat_mean.p50_ms_window, "p95_ms_window": lat_mean.p95_ms_window,
    "p50_round_sd": lat_spread,
    # median over seeds: the first CUDA run of a session pays one-time library workspace, so a max would charge it to one arm
    "peak_gpu_mb": g.peak_gpu_mb.median(), "rss_delta_mb_mean": g.rss_delta_mb.mean(),
}).reindex(ARMS)
SUMM.round(4).to_csv(RESULTS / f"summary_{TAG}.csv")

ref = RUNS[RUNS.arm == REF].set_index("seed")[metric]
cmp_rows = []
for arm in [a for a in ARMS if a != REF]:
    x = RUNS[RUNS.arm == arm].set_index("seed")[metric]
    d = x.values.mean() - ref.values.mean()
    vx, vr = x.var(ddof=1) / len(x), ref.var(ddof=1) / len(ref)
    se = np.sqrt(vx + vr)
    dfw = se**4 / (vx**2 / (len(x) - 1) + vr**2 / (len(ref) - 1))
    half = stats.t.ppf(0.975, dfw) * se
    common = x.index.intersection(ref.index)
    cmp_rows.append({
        "arm": arm, "reference": REF, "diff_mae_kw": d, "diff_pct_of_ref": 100 * d / ref.mean(),
        "welch_ci_lo": d - half, "welch_ci_hi": d + half,
        "welch_p": stats.ttest_ind(x, ref, equal_var=False).pvalue,
        "paired_p": stats.ttest_rel(x[common], ref[common]).pvalue,
        "effect_over_seed_sd": abs(d) / np.sqrt((x.var(ddof=1) + ref.var(ddof=1)) / 2)})
CMP = pd.DataFrame(cmp_rows)
CMP["welch_p_holm"] = holm(CMP.welch_p.values)
CMP["exceeds_noise"] = CMP.welch_p_holm < 0.05
CMP.round(5).to_csv(RESULTS / f"comparison_{TAG}.csv", index=False)

VERDICTS = []
for _, c in CMP.iterrows():
    if c.exceeds_noise:
        word = "better" if c.diff_mae_kw < 0 else "worse"
        VERDICTS.append(f"{c.arm} vs {REF}: EXCEEDS run-to-run noise ({c.diff_mae_kw:+.3f} kW, {c.diff_pct_of_ref:+.1f}%, "
                        f"{c.arm} {word}; Holm-adjusted p={c.welch_p_holm:.3g}; effect = {c.effect_over_seed_sd:.1f}x the seed SD).")
    else:
        VERDICTS.append(f"{c.arm} vs {REF}: NOT distinguishable from noise ({c.diff_mae_kw:+.3f} kW, {c.diff_pct_of_ref:+.1f}%; "
                        f"Welch 95% CI [{c.welch_ci_lo:+.3f}, {c.welch_ci_hi:+.3f}] includes 0; Holm-adjusted p={c.welch_p_holm:.3g}).")
for v in VERDICTS:
    print("-", v)


# Step 10: Tables and Plots
def md_table(df, digits=3, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)])


md = [f"# Results ({CFG['name']}, tag `{TAG}`)", "",
      f"Metric: test MAE in kW (lower is better), {int(SUMM.n_seeds.min())} seeds per arm. Reference arm: `{REF}`. "
      f"'Exceeds noise' = Welch p < 0.05 after Holm correction across {len(CMP)} comparisons.", "",
      "## Per-arm results", "", md_table(SUMM), "", "## Each arm vs reference", "", md_table(CMP, 4, index=False), "",
      f"No-learning references: same hours yesterday MAE = {NAIVE['naive_day_mae_kw']:.2f} kW; "
      f"same hours last week = {NAIVE['naive_week_mae_kw']:.2f} kW.", "", "## Verdict", ""]
md += [f"- {v}" for v in VERDICTS]
md += ["", f"Caveat: with {int(SUMM.n_seeds.min())} seeds per arm the test has low power; 'not distinguishable' means "
       "'not shown', not 'equal'. All seeds share one synthetic dataset, so the seed spread excludes data-sampling variance."]
(RESULTS / f"results_{TAG}.md").write_text("\n".join(md), encoding="utf-8")

colors = dict(zip(ARMS, ["#7a7a7a", "#1f77b4", "#d95f02", "#1b9e77"]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, a in enumerate(ARMS):
    v = RUNS[RUNS.arm == a][metric].values
    ax[0].scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, color=colors[a], alpha=.8, zorder=3)
    ax[0].errorbar(i, v.mean(), yerr=ci95(v), color="k", capsize=6, lw=2, zorder=4)
ax[0].set_xticks(range(len(ARMS)), ARMS)
ax[0].set_ylabel("test MAE (kW)")
ax[0].set_title("Every seed (dots), mean +/- 95% CI (bar)")
ax[0].axhline(NAIVE["naive_day_mae_kw"], ls="--", c="grey", lw=1)
ax[0].text(len(ARMS) - .6, NAIVE["naive_day_mae_kw"], "yesterday-naive", ha="right", va="bottom", fontsize=8, color="grey")
for a in ARMS:
    ax[1].scatter(SUMM.loc[a, "p50_ms_window"], SUMM.loc[a, "mae_mean"], color=colors[a], s=60, label=a)
ax[1].set_xlabel("inference latency (ms / window, p50 over interleaved rounds)")
ax[1].set_ylabel("mean test MAE (kW)")
ax[1].set_title("Accuracy vs cost")
ax[1].legend()
fig.tight_layout()
fig.savefig(RESULTS / f"mae_by_arm_{TAG}.png", dpi=150)
if IN_NOTEBOOK:
    plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5))
for i, c in CMP.reset_index(drop=True).iterrows():
    ax.errorbar(c.diff_mae_kw, i, xerr=[[c.diff_mae_kw - c.welch_ci_lo], [c.welch_ci_hi - c.diff_mae_kw]],
                fmt="o", color=colors[c.arm], capsize=5)
ax.axvline(0, c="k", lw=1)
ax.set_yticks(range(len(CMP)), CMP.arm)
ax.set_xlabel(f"MAE difference vs {REF} (kW), Welch 95% CI; <0 = better")
fig.tight_layout()
fig.savefig(RESULTS / f"diff_vs_reference_{TAG}.png", dpi=150)
if IN_NOTEBOOK:
    plt.show()
plt.close(fig)


# Step 11: Results JSON and Manifest
total_s = time.perf_counter() - T_START
MANIFEST = RESULTS / f"manifest_{TAG}.json"
if MANIFEST.exists() and not FRESH:  # a resumed invocation adds to the runtime already recorded
    total_s += json.loads(MANIFEST.read_text(encoding="utf-8")).get("total_runtime_s", 0.0)
arms_json = {a: {k: (None if (isinstance(v, float) and math.isnan(v)) else (v.item() if hasattr(v, "item") else v))
                 for k, v in SUMM.loc[a].items()} for a in ARMS}
(RESULTS / f"results_{TAG}.json").write_text(json.dumps({
    "tag": TAG, "reference_arm": REF, "seeds": SEEDS, "arms": arms_json,
    "comparison": json.loads(CMP.to_json(orient="records")), "verdicts": VERDICTS, "naive": NAIVE,
    "reference_checks": CHECKS}, indent=1), encoding="utf-8")
MANIFEST.write_text(json.dumps({
    "hardware": hardware_snapshot(ENV), "total_runtime_s": total_s, "data_sha256": DATA_SHA,
    "cpu_threads_pinned": CFG["threads"], "quick": QUICK, "config": CFG}, indent=1, default=str), encoding="utf-8")
print(f"wrote results/*_{TAG}.* in {total_s:.0f}s total")
