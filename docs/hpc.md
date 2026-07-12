# Running on HPC (NHR@FAU / Alex)

This document describes how to run NEUROSEG training on FAU's high-performance
computing systems, operated by **NHR@FAU**. It targets the **Alex** GPU cluster,
which is the right machine for the JEPA training in this project.

All commands and hostnames below reflect the NHR@FAU documentation at the time of
writing. NHR@FAU is the authoritative source and its setup can change — verify
against <https://doc.nhr.fau.de> before relying on a detail. Links to the exact
pages are collected at the bottom.

---

## Which cluster

| Cluster | Hardware | Use for NEUROSEG |
| ------- | -------- | ---------------- |
| **Alex** | NVIDIA **A40 48 GB** and **A100 40/80 GB** GPUs, AMD Milan CPUs | Yes — all JEPA training (H1/H2/H3) |
| Fritz | CPU-only parallel cluster | No (no GPU) |
| Woody / TinyGPU | throughput / entry GPU | Possible for small jobs, but Alex is preferred |

Alex is a **single-GPU-friendly** fit for this project: NEUROSEG training is not
distributed, so one A40 or A100 per job is enough — and both are far larger than
the 15 GB budget the config was originally tuned for.

---

## Access

- A basic FAU HPC account does **not** include Alex. Access is requested
  separately with a short justification of the compute need
  (<https://hpc.fau.de/tier3-access-to-alex/>).
- HPC-portal accounts have **no password** — authentication is **SSH key only**.
  Upload your public SSH key in the HPC portal before the first login.
- Compute is billed against a project quota (NPL). Do not leave interactive GPU
  sessions idle.

---

## Mental model

Three things trip up first-time HPC users:

1. **Login node vs compute node.** You SSH into a login node (`alex.nhr.fau.de`)
   that has **no GPU**. It is for editing, installing, and submitting jobs only.
   The GPUs live on compute nodes, reachable only through the **Slurm** scheduler
   (`sbatch` / `salloc`). Never run training on the login node.
2. **Filesystems.**
   - `$HOME` — small, backed up. Keep dotfiles/config here, not data.
   - `$WORK` — large, **not** backed up. Put code, datasets, and outputs here.
   - `$TMPDIR` — fast node-local NVMe SSD (7 TB on a40, 14 TB on a100), exists
     **only for the duration of a job**. Ideal for staging data.
3. **No internet on compute nodes.** Downloads (`pip`, dataset fetches) fail
   inside a job unless you set the HTTP proxy:
   ```bash
   export https_proxy=http://proxy.nhr.fau.de:80
   export http_proxy=http://proxy.nhr.fau.de:80
   ```
   The login node has internet, so one-time installs there need no proxy.

---

## One-time setup (on a login node)

```bash
ssh <user>@alex.nhr.fau.de

module load python                       # provides conda
# store conda packages/envs on $WORK so they don't fill the small $HOME
conda config --add pkgs_dirs $WORK/software/conda/pkgs
conda config --add envs_dirs $WORK/software/conda/envs

conda create -n neuroseg python=3.11 -y
conda activate neuroseg

cd $WORK
git clone https://github.com/mo-karbalaee/NEUROSEG.git
cd NEUROSEG
pip install -e .
```

Notes:
- `pip install` **inside an activated conda env** is the recommended pattern on
  NHR@FAU (their caution is against system-wide `pip --user`, not env-local pip).
- The default PyTorch CUDA wheels work on A40/A100. If `pip install -e .` yields a
  CPU-only torch, rebuild the environment from inside an interactive GPU job (see
  below) with the proxy set, so CUDA is autodetected.

Stage the data onto `$WORK` from your laptop:

```bash
rsync -av ~/neuroseg-labeled/ <user>@alex.nhr.fau.de:'$WORK/data/neuroseg-labeled/'
```

---

## Test interactively first

Before submitting a long batch job, grab a GPU for an hour and confirm the run
starts and one epoch progresses:

```bash
salloc --gres=gpu:a40:1 --time=1:00:00
module load python
conda activate neuroseg
cd $WORK/NEUROSEG
python main.py --mode train --H1 \
    --data $WORK/data/neuroseg-labeled \
    --labeled-data $WORK/data/neuroseg-labeled/neurofinder.00.00/neurofinder.00.00 \
    --output $WORK/neuroseg-output --config config.yaml
```

---

## Batch job script

Save as `h1.slurm` in `$WORK/NEUROSEG`:

```bash
#!/bin/bash -l
#SBATCH --job-name=neuroseg-h1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
conda activate neuroseg

cd $WORK/NEUROSEG
python main.py \
    --mode train --H1 \
    --data        $WORK/data/neuroseg-labeled \
    --labeled-data $WORK/data/neuroseg-labeled/neurofinder.00.00/neurofinder.00.00 \
    --output      $WORK/neuroseg-output \
    --config      config.yaml
```

- `#SBATCH --export=NONE` + `unset SLURM_EXPORT_ENV` is NHR@FAU's recommended
  clean-environment boilerplate (ensures Slurm variables and loaded modules
  propagate correctly).
- Request an A100 instead with `--gres=gpu:a100:1`, and add `-C a100_80` or
  `-C a100_40` to pin the memory variant.
- No proxy is needed in the job because the environment and data are already on
  `$WORK`.

For H2/H3, swap the flag and paths, following the CLI in the
[experiments doc](experiments.md):

```bash
# H2
python main.py --mode train --H2 \
    --data $WORK/data/neuroseg-labeled \
    --source-data $WORK/data/neuroseg-labeled/neurofinder.04.00/neurofinder.04.00 \
    --target-data $WORK/data/neuroseg-labeled/neurofinder.00.00/neurofinder.00.00 \
    --output $WORK/neuroseg-output --config config.yaml
```

---

## Submit, monitor, collect

```bash
sbatch h1.slurm                # prints a job ID; stdout goes to slurm-<id>.out
squeue -u $USER                # list your queued/running jobs
tail -f slurm-<jobid>.out      # watch tqdm progress live
scancel <jobid>                # cancel a job
```

Results (checkpoints, `logs/runs.csv`, figures) land in `$WORK/neuroseg-output`.
Copy them back to your laptop to run inference / inspect segmentation:

```bash
rsync -av <user>@alex.nhr.fau.de:'$WORK/neuroseg-output/' ~/neuroseg-output/
```

---

## Performance tips for this project

- **Scale up the config.** A40/A100 have far more memory than the 15 GB the config
  targets. Raise `batch_size` (e.g. 16–32) and optionally `img_size` in
  `config.yaml` for faster, better training. The config is environment-agnostic —
  edit it and resubmit.
- **Stage data to `$TMPDIR`.** `NeurofinderDataset` reads thousands of small TIFFs
  and loads every frame into RAM at start-up; the node-local SSD is much faster
  than `$WORK`. Copy at job start and point `--data` there:
  ```bash
  cp -r $WORK/data/neuroseg-labeled $TMPDIR/
  # ... --data $TMPDIR/neuroseg-labeled ...
  ```
- **Recursive dataset discovery** means `--data` can point at the dataset root and
  all nested `neurofinder.XX.XX/` sets are pooled for unlabeled pretraining. See
  the [data doc](data.md).
- **Walltime.** Alex allows much longer jobs than a Kaggle session, so a full
  pretrain + labeled-fraction sweep can run in one job. Size `--time` to the
  measured per-epoch cost.

---

## Sources

- [Alex cluster — NHR@FAU](https://doc.nhr.fau.de/clusters/alex/)
- [Slurm batch system — NHR@FAU](https://doc.nhr.fau.de/batch-processing/batch_system_slurm/)
- [Slurm job script examples — NHR@FAU](https://doc.nhr.fau.de/batch-processing/job-script-examples-slurm/)
- [Python environments / Conda — NHR@FAU](https://doc.nhr.fau.de/environment/python-env/)
- [Using node-local SSDs — NHR@FAU](https://doc.nhr.fau.de/data/staging/)
- [Tier3 access to Alex — NHR@FAU](https://hpc.fau.de/tier3-access-to-alex/)
