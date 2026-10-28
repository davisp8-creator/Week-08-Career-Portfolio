# Audit of the original submission

Scope: the Week 7 package this repository was refined from. Each finding says what a stranger could not have reproduced or checked, and what was changed. Findings were checked against the committed raw files, not recalled.

## What a stranger could not reproduce or verify

| # | Finding | Evidence | Status |
|---|---|---|---|
| 1 | **No installable specification for the GPU result.** `requirements.txt` pinned only the CPU environment (torch 2.14.1, numpy 2.5.4, pandas 3.1.0). The T4 result was produced under torch 2.11.0+cu130, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, matplotlib 3.10.0, Python 3.13.16, which appeared only inside `manifest_t4gpu.json`. The file also told readers not to install it on Colab. | `requirements.txt` header; `results/manifest_t4gpu.json` | Fixed: `requirements-t4.txt` and `requirements-cpu.txt`. One gap remains: the psutil version on the T4 session was never recorded. |
| 2 | **The entry point installed unpinned packages at run time.** `env_setup.py` pip-installed any missing import with no version, so a clean machine silently got whatever was newest. | `env_setup.py`, `install_missing` | Fixed: it now stops and points to the pinned files unless `ALLOW_UNPINNED_INSTALL=1` (Colab excepted, where the preinstalled stack is the reference and the manifest records it). |
| 3 | **Hardware and software stack are confounded in the CPU-vs-T4 comparison.** The two tags differ in device, Python (3.14.7 vs 3.13.16), torch (2.14.1 vs 2.11.0), numpy, pandas and host. Any latency or training-time gap between them cannot be attributed to the GPU alone. | the two manifests | Not fixable without another machine. Stated in README limitation 3. Within-tag comparisons (arm vs arm) are unaffected. |
| 4 | **The headline was never repeated on the GPU.** One T4 session produced every GPU number. The Annex draft says "repeat the T4 session before quoting latency outside the company" and marks it open. | Annex IV doc §3.2 | Fixed: `scripts/repeat_study.py` plus a second T4 session. Five full runs: per-run test MAE bit-identical (max difference 0.0 kW), latency spread at most 0.0004 ms within a session and +2.4% between sessions. Results in `results/repeats/t4gpu/` and the README. The repeats ran back to back on one VM, so they do not cover other GPUs or days. |
| 5 | **A reproducibility claim had no retained evidence.** The Annex draft says the study "was run three times locally ... every test MAE identical to three decimals". Only one CPU `runs_cpu.csv` is committed, so the claim cannot be checked. The README also promised bit-identical numbers on the same machine, which no committed artifact demonstrates. | `docs/ANNEX_IV_TECHNICAL_DOCUMENTATION.md` §2.2; `results/` | Annex now carries an audit note. The harness records `max |ΔMAE|` between repeats, so the claim became a measured number on the T4: bit-identical across five full runs. The local-CPU claim stays unretained. |
| 6 | **"No difference" was reported without the size of difference the study could have detected.** Five seeds were run, but the text did not say what 5 seeds could resolve. | `results_t4gpu.md` caveat is qualitative only | Fixed: `scripts/power_analysis.py` computes the minimum detectable difference from the committed per-seed data. |
| 7 | **Precision and run shape were not in the results footnote.** Batch sizes, sequence length and precision lived in the config or code, not beside the numbers. Precision is fp32 throughout (checked in code: `torch.float32`, no autocast or half precision). | `configs/ablation.yaml`; `scripts/run_ablation.py` | Fixed in the README hardware footnote. |
| 8 | **Course scaffolding obscured the artifact.** Colab bootstrap cell, a notebook generator, OS installers, an optional "push results to README with a token" step, and a scenario naming a manager and a compliance lead. | original repo layout | Removed from the entry point and repository; the scenario is labelled fictional. |

## What was already sound (kept)

- Data are generated from a fixed seed and hashed (SHA-256 `2542d60bc3f5…`, identical on both machines despite different numpy and pandas versions).
- One factor changes between arms; seeds are the outer loop and arm order inside a seed is shuffled; batch order depends on the seed alone.
- Latency is timed in interleaved rounds with warm-up discarded and `cuda.synchronize()` around each forward.
- Memory is reported as a median over seeds after the first CUDA run was seen to pay a one-time workspace cost (593 MB vs 528 MB).
- Per-run raw results (`runs_<tag>.csv`) are committed, so every table and plot regenerates from raw rows.

## Checks run while preparing this audit

- Re-derived the Annex IV cross-hardware statements from `runs_cpu.csv` and `runs_t4gpu.csv`: per-run |T4 − CPU| test MAE is at most 0.169 kW for the three working arms and 1.059 kW for `none`, as stated.
- Regenerated `summary_cpu.csv`, `comparison_cpu.csv` and `results_cpu.md` from the committed `runs_cpu.csv` with the modified entry point: byte-identical to the committed files.
- Smoke-tested `repeat_study.py` on the CPU with the reduced `quick:` config before the T4 run: per-run test MAEs were bit-identical. That tested the harness only.
- Reviewed the T4 run after it finished: the second-session main run and repeats 0–2 match the first session's `runs_t4gpu.csv` exactly on test MAE (max difference 0.0 kW), and every manifest shows the same torch, Python, driver and data hash.
