# =============================================================================
# region SCRIPT DETAILS
# =============================================================================
# Single-GPU Practitioner Portfolio
# Positional-Encoding Ablation: README results writer
# Objective: Write the local-CPU vs Colab-T4 measurements and the repeat-run summary into README.md and,
# optionally, push them to GitHub with a fine-grained personal access token.
#
# Usage:
#   python scripts/update_readme.py            # re-render the README results section
#   python scripts/update_readme.py --push     # also commit + push (prompts for a token)
#   python scripts/update_readme.py --push --tag t4gpu
# The last cell of the Colab notebook calls offer_publish(), which asks first.
# =============================================================================
# endregion
# =============================================================================
# Step 1: Render the Results Section
# Reads results/results_<tag>.json + manifest_<tag>.json for every full-run tag present
# (cpu = local machine, t4gpu = Colab T4, gpu = any other GPU) and renders the hardware table,
# the per-arm accuracy and cost tables, the verdict, and the repeat-run summary (results/repeats/<tag>) between the README markers.
# =============================================================================
"""
https://docs.python.org/3/library/json.html
https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables
"""

import argparse
import base64
import getpass
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
TAG_ORDER = ["cpu", "t4gpu", "gpu"]
HW_LABEL = {"cpu": "Local CPU", "t4gpu": "Colab T4 GPU", "gpu": "Other GPU"}
TAG_LABEL = HW_LABEL


def load_runs(root):
    runs = {}
    for tag in TAG_ORDER:
        r, m = root / "results" / f"results_{tag}.json", root / "results" / f"manifest_{tag}.json"
        if r.exists() and m.exists():
            runs[tag] = (json.loads(r.read_text(encoding="utf-8")), json.loads(m.read_text(encoding="utf-8")))
    return runs


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def fmt(x, kind):
    x = _num(x)
    if x is None:
        return "n/a"
    return {"p": f"{x:.3f}", "ms": f"{x:.3f} ms", "mb": f"{x:,.2f} MB", "s": f"{x:,.1f} s", "int": f"{int(x):,}"}[kind]


def device_label(m):
    h = m["hardware"]
    if h.get("gpu_name"):
        return f"{h['gpu_name']} ({h['vram_gb']} GB, sm_{str(h['compute_capability']).replace('.', '')})"
    vm = " (virtual machine)" if h.get("virtualized") else ""
    return f"CPU only: {h['cpu_model']}, {h['cpu_count']} logical cores{vm}"


def arm_table(res):
    head = ["| Arm | Params | Test MAE, kW (mean ± seed SD) | Range | p50 latency, ms/window | Train s (mean) | Peak GPU MB above floor (median over seeds) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, c in res["arms"].items():
        head.append(f"| `{arm}` | {int(c['n_params']):,} | {c['mae_mean']:.3f} ± {c['mae_sd']:.3f} | "
                    f"{c['mae_min']:.2f}–{c['mae_max']:.2f} | {fmt(c['p50_ms_window'], 'p')} | {fmt(c['train_s_mean'], 's')} | "
                    f"{fmt(c['peak_gpu_mb'], 'p')} |")
    return head


def repeat_section(root, tag):
    base = root / "results" / "repeats" / tag
    sj, sc = base / "summary.json", base / "summary.csv"
    if not (sj.exists() and sc.exists()):
        return []
    meta = json.loads(sj.read_text(encoding="utf-8"))
    rows = [ln.split(",") for ln in sc.read_text(encoding="utf-8").strip().splitlines()]
    head, data = rows[0], rows[1:]
    ix = {h: i for i, h in enumerate(head)}
    out = [f"### Repeat runs on `{tag}`: the whole study retrained {meta['repeats']} times", "",
           "| Arm | Test MAE, kW (mean of repeat means) | Spread of repeat means, kW | p50 latency, ms/window | Latency spread, ms | Train s per run | Train-time spread, s |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in data:
        out.append(f"| `{r[ix['arm']]}` | {float(r[ix['mae_mean_of_means']]):.3f} | {float(r[ix['mae_mean_range_across_repeats']]):.3f} | "
                   f"{float(r[ix['p50_ms_mean']]):.4f} | {float(r[ix['p50_ms_range']]):.4f} | "
                   f"{float(r[ix['train_s_mean']]):.1f} | {float(r[ix['train_s_range']]):.1f} |")
    verdict = "bit-identical" if meta["bit_identical_across_repeats"] else "not bit-identical"
    out += ["", f"Largest per-run test-MAE difference between repeats: {meta['max_abs_diff_per_run_test_mae_kw']:.3e} kW ({verdict}).", ""]
    return out


def render_section(root=ROOT):
    runs = load_runs(root)
    if not runs:
        return f"{BEGIN}\n_No full-run results in `results/` yet._\n{END}"
    tags = list(runs)
    head = "| | " + " | ".join(HW_LABEL[t] for t in tags) + " |"
    sep = "|---|" + "---:|" * len(tags)
    hw = [head, sep,
          "| Device | " + " | ".join(device_label(runs[t][1]) for t in tags) + " |",
          "| Host class | " + " | ".join(f"`{runs[t][1]['hardware']['host_class']}`" for t in tags) + " |",
          "| Platform / Python | " + " | ".join(f"{runs[t][1]['hardware']['platform']} / {runs[t][1]['hardware']['python']}" for t in tags) + " |",
          "| torch | " + " | ".join(f"`{runs[t][1]['hardware']['torch']}`" for t in tags) + " |",
          "| Total script runtime (cumulative) | " + " | ".join(fmt(runs[t][1]["total_runtime_s"], "s") for t in tags) + " |",
          "| Data SHA-256 | " + " | ".join(f"`{runs[t][1]['data_sha256'][:12]}…`" for t in tags) + " |"]
    body = []
    for t in tags:
        res = runs[t][0]
        body += [f"### {HW_LABEL[t]} (`{t}`): {len(res['seeds'])} seeds per arm, reference arm `{res['reference_arm']}`", "",
                 *arm_table(res), "",
                 f"No-learning references: yesterday-naive {res['naive']['naive_day_mae_kw']:.2f} kW, "
                 f"last-week-naive {res['naive']['naive_week_mae_kw']:.2f} kW.", "",
                 *[f"- {v}" for v in res["verdicts"]], ""]
        fig = root / "results" / f"mae_by_arm_{t}.png"
        if fig.exists():
            body += [f"![Test MAE by arm, every seed](results/{fig.name})", ""]
        body += repeat_section(root, t)
    note = "" if len(tags) > 1 else ("\n> Only one machine is in `results/` so far. The other column appears after you run it "
                                    "(installer locally, the Colab notebook for the T4).\n")
    return "\n".join([
        BEGIN,
        "<!-- Generated by scripts/update_readme.py from results/*.json; edits inside these markers are overwritten. -->",
        "## Measured on each machine (generated)",
        note,
        "Same code, seeds and budgets on every machine. **Accuracy should agree across hardware** up to floating-point order; "
        "**latency and memory are what the hardware changes.** 'Exceeds noise' means the Holm-adjusted Welch p-value is below 0.05. "
        "'Not distinguishable' means not shown at this seed count, not equal.",
        "",
        "### Hardware", "", *hw, "",
        *body,
        END])


def write_readme(root=ROOT):
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"README.md has no {BEGIN} ... {END} markers; add them where the results should go.")
    new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: render_section(root), text, flags=re.S)
    changed = new != text
    if changed:
        path.write_text(new, encoding="utf-8")
    return changed


# =============================================================================
# Step 2: Commit and Push with a Fine-Grained Token
# The token comes from the Colab secret GITHUB_TOKEN if it exists, otherwise from a hidden
# getpass prompt (blank = use this machine's existing git credentials). It is sent per command
# as an HTTP auth header: never printed, never written to .git/config, never put in the remote
# URL. Results commit first, then a rebase pull (our results win on conflict), then the README
# is re-rendered so the comparison includes the latest results from GitHub, then push.
# =============================================================================
"""
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token
https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpextraHeader
https://git-scm.com/docs/git-pull#Documentation/git-pull.txt---rebase
https://docs.python.org/3/library/getpass.html
"""


class GitError(RuntimeError):
    pass


def _git(root, *args, token=None, check=True):
    cmd = ["git"]
    basic = None
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]
    out = subprocess.run(cmd + list(args), cwd=root, capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    if token:
        text = text.replace(token, "***").replace(basic, "***")
    if check and out.returncode != 0:
        raise GitError(f"git {args[0]} failed:\n{text}")
    return out.returncode, text


def colab_secret_token():
    try:
        from google.colab import userdata

        return userdata.get("GITHUB_TOKEN") or None
    except Exception:  # not on Colab, secret missing, or notebook access not granted
        return None


def get_token(ask_secret=getpass.getpass):
    token = colab_secret_token()
    if token:
        print("[publish] using the Colab secret GITHUB_TOKEN")
        return token
    print("[publish] Fine-grained token: this repo only, Repository permissions > Contents: Read and write.")
    token = ask_secret("[publish] GitHub token (hidden; blank = use existing git credentials): ").strip()
    return token or None


def ensure_identity(root, ask=input):
    for key, prompt in (("user.name", "Your name for the commit: "), ("user.email", "Your email for the commit: ")):
        if not _git(root, "config", key, check=False)[1]:
            _git(root, "config", key, ask(f"[publish] {prompt}").strip())


def require_clone(root):
    code, top = _git(root, "rev-parse", "--show-toplevel", check=False)
    if code != 0 or Path(top).resolve() != Path(root).resolve():
        raise GitError(f"{root} is not the top of a git clone. On Colab, run the notebook's Colab Bootstrap cell first (COLAB.md).")


def push_results(root, tag, token=None, ask=input):
    require_clone(root)
    ensure_identity(root, ask)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")[1]
    names = ("results_{t}.json", "manifest_{t}.json", "runs_{t}.csv", "latency_runs_{t}.csv", "summary_{t}.csv",
             "comparison_{t}.csv", "results_{t}.md", "mae_by_arm_{t}.png", "diff_vs_reference_{t}.png",
             "power_{t}.csv", "power_{t}.json", "power_{t}.png")
    result_paths = [p for p in [f"results/{n.format(t=tag)}" for n in names] if (root / p).exists()]
    if (root / "results" / "repeats" / tag).is_dir():
        result_paths.append(f"results/repeats/{tag}")
    _git(root, "checkout", "--", "README.md")  # regenerated after the pull, so drop the local render
    _git(root, "add", "--", *result_paths)
    if _git(root, "diff", "--cached", "--quiet", check=False)[0]:
        _git(root, "commit", "-m", f"Add {TAG_LABEL.get(tag, tag)} results ({tag})")
    print(f"[publish] pulling origin/{branch} ...")
    code, text = _git(root, "pull", "--rebase", "-X", "theirs", "origin", branch, token=token, check=False)
    if code != 0:
        _git(root, "rebase", "--abort", check=False)
        raise GitError(f"pull --rebase failed, nothing was pushed:\n{text}")
    write_readme(root)
    _git(root, "add", "README.md")
    if _git(root, "diff", "--cached", "--quiet", check=False)[0]:
        _git(root, "commit", "-m", "Update README: measured results by machine and repeat runs")
    print(f"[publish] pushing to origin/{branch} ...")
    _git(root, "push", "origin", f"HEAD:{branch}", token=token)
    print(f"[publish] done: README.md and {len(result_paths)} result path(s) pushed to origin/{branch}")


# =============================================================================
# Step 3: Prompt at the End of a Run
# Called by the notebook's last step: asks before touching README.md and asks again before
# pushing. Non-interactive runs only print how to do it later.
# =============================================================================


def _yes(answer):
    return answer.strip().lower() in ("y", "yes")


def offer_publish(root, tag, interactive, ask=input, ask_secret=getpass.getpass):
    later = "python scripts/update_readme.py --push"
    if not interactive:
        print(f"[publish] non-interactive run: to write README.md and push later, run  {later}")
        return
    try:
        if not _yes(ask("Write these results into README.md (measured results by machine, repeat runs)? [y/N] ")):
            print(f"[publish] skipped. Later: {later}")
            return
        changed = write_readme(root)
        print(f"[publish] README.md {'updated' if changed else 'already up to date'} (results section only)")
        if not _yes(ask(f"Commit and push README.md + results/*_{tag}* to GitHub? [y/N] ")):
            print("[publish] not pushed. README.md is updated locally only.")
            return
        require_clone(root)  # fail before asking for a token that could not be used
        push_results(root, tag, token=get_token(ask_secret), ask=ask)
    except (EOFError, KeyboardInterrupt):
        print(f"\n[publish] cancelled. Later: {later}")
    except GitError as err:
        print(f"[publish] {err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render the README results section; optionally commit and push.")
    ap.add_argument("--push", action="store_true", help="commit results + README and push (prompts for a token)")
    ap.add_argument("--tag", default=None, help="results tag to commit (default: t4gpu if present, else cpu)")
    args = ap.parse_args()
    if args.push:
        runs = load_runs(ROOT)
        tag = args.tag or ("t4gpu" if "t4gpu" in runs else "cpu")
        try:
            require_clone(ROOT)
            push_results(ROOT, tag, token=get_token())
        except GitError as err:
            sys.exit(f"[publish] {err}")
    else:
        print(f"README.md {'updated' if write_readme(ROOT) else 'already up to date'}")
