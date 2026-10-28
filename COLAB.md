# Running the notebook on a Colab T4 GPU

The notebook `Assignment_08_Portfolio_Colab.ipynb` produces the T4 measurements for this portfolio: a fresh 20-run study, a power analysis, and three whole-study repeats. At the end it offers to write the results into `README.md` and push them to GitHub with a fine-grained token. Local CPU results (`results/*_cpu.*`) are never overwritten.

## 1. One-time: push the repo and create a token

1. The repository at https://github.com/davisp8-creator/Week-08-Career-Portfolio must contain the whole folder, including `Assignment_08_Portfolio_Colab.ipynb`. Colab clones it, so anything not pushed does not exist there.
2. GitHub > Settings > Developer settings > Personal access tokens > **Fine-grained tokens** > Generate new token:
   - **Repository access:** only select repositories > `Week-08-Career-Portfolio`
   - **Permissions:** Repository permissions > **Contents: Read and write** (nothing else)
   - **Expiration:** short (7 to 30 days); revoke it when the assignment is done
3. In Colab open the **Secrets** panel (key icon, left sidebar), add a secret named `GITHUB_TOKEN` with the token as its value, and turn on *Notebook access*. The notebook then uses it for the clone (needed only if the repo is private) and for the push, and never asks you to paste it. Without the secret you get a hidden prompt at the end.

## 2. Runtime and notebook

**Runtime > Change runtime type > T4 GPU > Save.** Open the notebook from GitHub (File > Open notebook > GitHub tab, paste the repo URL) or upload the `.ipynb` file. The **Colab Bootstrap** cell clones the repo once and changes into it. It is safe to re-run: when `env_setup.py` already exists in the working directory it does nothing, so **Run all** never clones twice or nests folders. Do not add your own `!git clone` and `%cd` cell. If folders already nested, use **Runtime > Disconnect and delete runtime**, then Run all again.

The clone URL is written into the notebook when `python scripts/build_notebook.py --repo-url <url>` runs (or from the `origin` remote when the folder is a git clone). Rebuild and push the notebook after any change to `scripts/run_ablation.py`.

## 3. Run all

**Runtime > Run all.** The **Run Settings** cell sets `RETRAIN = True`: all 20 runs retrain on this T4, so `runs_t4gpu.csv` and `manifest_t4gpu.json` come from the same session. Step 1 should print:

```text
[setup] host=colab-t4 (Colab (T4 GPU))  platform=colab  python=3.x.x  torch=2.x.x+cu1xx  device=cuda  gpu=Tesla T4 (14.x GB, sm_75)  results tag='t4gpu'
```

Check for `host=colab-t4`, `gpu=Tesla T4` and `results tag='t4gpu'`.

| Message | Meaning | Fix |
|---|---|---|
| `Colab is on a CPU-only runtime` | The runtime type was not changed | Change runtime type to T4 GPU, then Run all again |
| `nvidia-smi sees a GPU ... but torch cannot use it` | A CPU torch was installed over Colab's CUDA build | Runtime > Disconnect and delete runtime, then run again without installing anything |
| `GPU is <name>, not a T4` | Colab assigned a different GPU | Results are tagged `gpu`, so the T4 files are not overwritten; the push step then uses that tag |
| `FileNotFoundError: configs/ablation.yaml` | The Colab Bootstrap cell was skipped or deleted | Restore it by rebuilding the notebook, or Disconnect and delete runtime, then Run all |

Do not run `pip install -r requirements-t4.txt` on Colab. It does not pin torch on purpose, because Colab ships its own CUDA build, and reinstalling torch could hide the T4 from it. The Colab library versions will differ from `requirements-t4.txt`; each run records the versions it used in `results/manifest_t4gpu.json` and in every `results/repeats/t4gpu/rep<k>/manifest_t4gpu.json`.

Time: the first T4 session took 492 s for the whole study. Expect about 8 minutes for the main run and about 8 minutes per repeat, so roughly 35 minutes in total. Quote the figure from your own session's manifest, not from this estimate. Progress is saved after every run. If the session drops during the repeats, reconnect, Run all again with `RETRAIN = False`, and finished repeats are skipped.

## 4. The end-of-run prompts (Step 14)

```text
Write these results into README.md (measured results by machine, repeat runs)? [y/N] y
[publish] README.md updated (results section only)
Commit and push README.md + results/*_t4gpu* to GitHub? [y/N] y
[publish] using the Colab secret GITHUB_TOKEN
[publish] pulling origin/main ...
[publish] pushing to origin/main ...
```

What the push does, in order:

1. Commits the `*_t4gpu.*` result files, the power-analysis files and `results/repeats/t4gpu/`.
2. Pulls `origin/main` with a rebase, so a newer CPU run pushed from your machine comes in here.
3. Re-renders the generated results section of the README from every result now in the clone, and commits it.
4. Pushes.

The token is sent per command as an HTTP header. It is never printed, never written to `.git/config`, never added to the remote URL, and never saved in the notebook. Answer **N** to either question to skip; nothing is pushed without your second yes.

## 5. After the run

- The README keeps two kinds of results. The hand-written tables near the top report the first T4 session. The generated block ("Measured on each machine") reports the latest files in `results/`. After a successful run, read the repeat summary, then update the hand-written headline and the "Repeat-run status" section to match. Reword the RoPE cost claim if the latency spread between repeats changes it.
- Download nothing from Colab; the push is the record. If you skipped it, run `!python scripts/update_readme.py --push` in the same session.
- Re-render only, with no git: `python scripts/update_readme.py`.
- Revoke the token.

## What to expect

Accuracy should agree between machines up to floating-point order. Latency and memory are what the hardware changes. These models are small, so a T4 will not always beat a modern CPU on latency, because kernel-launch overhead dominates. The repeat summary reports whether per-run test MAEs are bit-identical between repeats on the T4; that is a measured result, so write down whichever way it comes out.
