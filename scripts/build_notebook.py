"""Build the Colab notebook and the simplified script from the full assignment script.

The full script follows the course layout: a SCRIPT DETAILS region, then one banner per
"Step N: Title" whose comment lines are the step's explanation and whose docstring lists
its references. This turns each step into a markdown cell (title, explanation,
references) plus a code cell, and writes a simplified script with the banners,
explanations and references stripped.

The Step 1 block between the INLINE-ENV-SETUP markers is replaced by env_setup.py's
source in the notebook, so the notebook runs on Colab with nothing else uploaded.

Usage:  python scripts/build_notebook.py [--repo-url https://github.com/<user>/<repo>]
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "run_ablation.py"
SIMPLIFIED = ROOT / "scripts" / "run_ablation_simplified.py"
ENV_SETUP = ROOT / "env_setup.py"
NOTEBOOK = ROOT / "Assignment_08_Portfolio_Colab.ipynb"
RULE = re.compile(r"^# =+$")
STEP = re.compile(r"^# Step (\d+): (.+)$")


def uncomment(lines):
    return [re.sub(r"^# ?", "", ln) for ln in lines]


def parse(text):
    """Split the script into (header_lines, header_refs, [step dicts])."""
    lines = text.split("\n")
    starts = [i for i, ln in enumerate(lines) if STEP.match(ln)]
    header = lines[: starts[0]]
    steps = []
    for k, s in enumerate(starts):
        end = starts[k + 1] - 1 if k + 1 < len(starts) else len(lines)
        num, title = STEP.match(lines[s]).groups()
        i = s + 1
        desc = []
        while not RULE.match(lines[i]):
            desc.append(lines[i])
            i += 1
        i += 1
        while i < end and not lines[i].strip():
            i += 1
        refs = []
        if i < end and lines[i] == '"""':
            i += 1
            while lines[i] != '"""':
                refs.append(lines[i].strip())
                i += 1
            i += 1
        body = lines[i:end]
        while body and (not body[-1].strip() or RULE.match(body[-1]) or body[-1] == "# endregion"):
            body.pop()
        steps.append(dict(num=int(num), title=title, desc=uncomment(desc),
                          refs=[r for r in refs if r], code="\n".join(body).strip("\n")))
    return header, steps


def header_markdown(header):
    """Course header + overview + APA references -> the notebook's title cell."""
    comments, refs, in_doc = [], [], False
    for ln in header:
        if ln == '"""':
            in_doc = not in_doc
        elif in_doc:
            refs.append(ln)
        elif ln.startswith("#") and not RULE.match(ln) and "region SCRIPT DETAILS" not in ln and ln.strip() != "# endregion":
            comments.append(ln)
    body = uncomment(comments)
    course, week, objective = body[0], body[1], body[2]
    entries = [" ".join(e.split()) for e in "\n".join(refs).split("\n\n") if e.strip()]
    return ([f"# {week}", f"*{course}*", "", f"**{objective}**"] + body[3:]
            + ["", "**References**", ""] + [f"- {e}" for e in entries])


def inline_env_setup(code):
    src = ENV_SETUP.read_text(encoding="utf-8")
    src = src[: src.index('\nif __name__ == "__main__":')]
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)  # drop module docstring
    inlined = ("# ── env_setup.py (inlined by scripts/build_notebook.py; edit the .py, then rebuild) ──\n"
               + src.strip() + "\n# ── end env_setup.py ──\n\nROOT = Path.cwd()")
    return re.sub(r"# Local runs import env_setup.*?# INLINE-ENV-SETUP-END", lambda _: inlined,
                  code, flags=re.S)


def cell(kind, lines):
    src = [ln + "\n" for ln in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    c = dict(cell_type=kind, metadata={}, source=src)
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def repo_url(cli_url=None):
    """The clone URL: the git remote when there is one, else --repo-url, else the REPO_URL environment variable."""
    try:
        out = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except FileNotFoundError:  # git is not installed
        pass
    return cli_url or os.environ.get("REPO_URL") or None


def bootstrap_cells(url):
    """Colab only has the uploaded notebook, so clone the repo once and move into it.
    Safe to re-run (Run all): skipped when already inside the clone, so it never nests."""
    code = [
        "# Colab bootstrap: clone the repo once and move into it. Safe to re-run: skipped inside the clone.",
        "import base64, os, subprocess",
        "",
        f'REPO_URL = "{url}"',
        'REPO_DIR = REPO_URL.rsplit("/", 1)[-1].removesuffix(".git")',
        "",
        "def colab_token():",
        "    try:",
        "        from google.colab import userdata",
        '        return userdata.get("GITHUB_TOKEN") or None',
        "    except Exception:  # not on Colab, secret missing, or notebook access not granted",
        "        return None",
        "",
        'if not os.path.exists("env_setup.py"):',
        "    if not os.path.isdir(REPO_DIR):",
        '        cmd = ["git"]',
        "        token = colab_token()  # only needed when the repo is private; sent as a header, never stored",
        "        if token:",
        '            basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()',
        '            cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]',
        '        subprocess.run(cmd + ["clone", "-q", REPO_URL], check=True)',
        "    os.chdir(REPO_DIR)",
        'print("working directory:", os.getcwd())',
    ]
    md = ["## Colab Bootstrap", "",
          "A Colab runtime starts with only this notebook. This cell clones the repo once and changes into it, so "
          "`scripts/` and `results/` exist. Running it again does nothing. A private repo clones with the `GITHUB_TOKEN` secret."]
    return [cell("markdown", md), cell("code", code)]


def zip_bootstrap_cells():
    """Fallback when the repo has no git remote: the user uploads the repo zip next to the notebook.
    Safe to re-run: skipped once env_setup.py is present in the working directory."""
    code = [
        "# Colab bootstrap (no GitHub remote): unpack the uploaded repo zip once and move into it. Safe to re-run.",
        "import glob, os, zipfile",
        "",
        'if not os.path.exists("env_setup.py"):',
        '    found = [d for d in glob.glob("*/") if os.path.exists(os.path.join(d, "env_setup.py"))]',
        "    if not found:",
        '        zips = sorted(glob.glob("Week-08*.zip"))',
        "        if not zips:",
        "            raise FileNotFoundError(",
        '                "Upload the Week-08 repo zip to /content (Files panel, left sidebar), then Run all again.")',
        '        zipfile.ZipFile(zips[0]).extractall(".")',
        '        found = [d for d in glob.glob("*/") if os.path.exists(os.path.join(d, "env_setup.py"))]',
        "    os.chdir(found[0])",
        'print("working directory:", os.getcwd())',
    ]
    md = ["## Colab Bootstrap", "",
          "A Colab runtime starts with only this notebook. Upload the repo zip to `/content` (Files panel in the left "
          "sidebar), then run this cell: it unpacks the zip once and changes into the folder, so `scripts/`, `configs/` "
          "and `results/` exist. Running it again does nothing."]
    return [cell("markdown", md), cell("code", code)]


def settings_cells():
    """One visible switch, set before Step 2 reads it. Retraining keeps each session's runs and manifest together."""
    md = ["## Run Settings", "",
          "`RETRAIN = True` retrains all 20 (arm, seed) runs on this machine, about 8 minutes on a T4, so `results/runs_<tag>.csv` "
          "and `manifest_<tag>.json` come from the same session. The committed T4 results from the first session are replaced "
          "in your clone (they stay in git history). Set it to `False` only to regenerate tables from rows already in `results/`; "
          "the manifest would then record this session's hardware beside older runs."]
    code = ["import os",
            "",
            "RETRAIN = True",
            'os.environ["ABLATE_FRESH"] = "1" if RETRAIN else "0"']
    return [cell("markdown", md), cell("code", code)]


def closing_cells(first_step):
    """Cells that sit beyond the script: power analysis, repeat study, publish. They call the same scripts a
    local user runs, so the notebook adds no logic of its own."""
    n = first_step
    power_md = [f"## Step {n}: Power Analysis", "",
                "Computes the smallest true difference the seed budget could detect, and the T4 time needed to resolve "
                "smaller ones, from the per-seed rows this session just wrote. Output: `results/power_<tag>.*`."]
    power = [
        "import subprocess, sys",
        "from IPython.display import Image, display",
        "",
        'subprocess.run([sys.executable, "scripts/power_analysis.py", "--tag", ENV["tag"]], check=True)',
        'display(Image("results/power_" + ENV["tag"] + ".png"))',
    ]
    rep_md = [f"## Step {n + 1}: Repeat the Whole Study", "",
              "Retrains every (arm, seed) run from scratch `REPEATS` times and reports how far arm means, latency and "
              "training time move between repeats, and whether per-run test MAEs are bit-identical. Each repeat is archived "
              "under `results/repeats/<tag>/rep<k>/`; the single-session results above are restored afterwards. "
              "About 8 minutes per repeat on a T4. A dropped session resumes: finished repeats are skipped."]
    rep = [
        "REPEATS = 3  # the assignment standard is at least three",
        'subprocess.run([sys.executable, "scripts/repeat_study.py", "--n", str(REPEATS)], check=True)',
    ]
    pub_md = [f"## Step {n + 2}: Write Results Back to GitHub", "",
              "Asks before touching `README.md`, then asks again before pushing. The token comes from the Colab secret "
              "`GITHUB_TOKEN` (Secrets panel, with Notebook access on) or a hidden prompt. It is sent per command as an HTTP "
              "header and is never printed, stored in `.git/config`, or placed in the remote URL. See `COLAB.md`."]
    pub = [
        'sys.path.insert(0, "scripts")',
        "from update_readme import offer_publish",
        "",
        'offer_publish(ROOT, ENV["tag"], interactive=True)',
    ]
    return [cell("markdown", power_md), cell("code", power), cell("markdown", rep_md), cell("code", rep),
            cell("markdown", pub_md), cell("code", pub)]


def build_notebook(header, steps, cli_url=None):
    cells = [cell("markdown", header_markdown(header))]
    url = repo_url(cli_url)
    if url:
        cells += bootstrap_cells(url)
    else:
        print("[build_notebook] no git remote and no --repo-url: using the upload-a-zip Colab bootstrap cell (see COLAB.md)")
        cells += zip_bootstrap_cells()
    cells += settings_cells()
    for st in steps:
        md = [f"## Step {st['num']}: {st['title']}", ""] + st["desc"]
        if st["refs"]:
            md += ["", "References: " + " · ".join(f"[{n}]({u})" for n, u in enumerate(st["refs"], 1))]
        cells.append(cell("markdown", md))
        if st["code"]:
            code = inline_env_setup(st["code"]) if "INLINE-ENV-SETUP-BEGIN" in st["code"] else st["code"]
            cells.append(cell("code", code.split("\n")))
    cells += closing_cells(steps[-1]["num"] + 1)
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(accelerator="GPU", colab=dict(gpuType="T4", provenance=[]),
                            kernelspec=dict(display_name="Python 3", name="python3"),
                            language_info=dict(name="python")))
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(cells)


def build_simplified(steps):
    parts = []
    for st in steps:
        if not st["code"]:
            continue
        code = re.sub(r"# Local runs import env_setup.*?\n# INLINE-ENV-SETUP-BEGIN\n", "", st["code"], flags=re.S)
        code = re.sub(r"\n*# INLINE-ENV-SETUP-END\n", "\n", code)
        parts.append(f"# Step {st['num']}: {st['title']}\n{code}")
    SIMPLIFIED.write_text("\n\n\n".join(parts) + "\n", encoding="utf-8")


def build(cli_url=None):
    header, steps = parse(SRC.read_text(encoding="utf-8"))
    n = build_notebook(header, steps, cli_url)
    build_simplified(steps)
    print(f"{len(steps)} steps -> {NOTEBOOK.name} ({n} cells), {SIMPLIFIED.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default=None, help="clone URL to embed when the folder has no git remote")
    build(ap.parse_args().repo_url)
