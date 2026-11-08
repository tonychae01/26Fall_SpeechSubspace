# Small Wingert Method Pilot

Use notebooks for explanation and inspection; keep repeatable preparation and batch training here.
Run commands from the repository root.

## Environment

The isolated Python 3.10 environment is at `/clusterfs/bebb/users/tonychae01/envs/env-speech`.
TensorFlow 2.15.1 uses Keras 2, which matches the NEMS layer API used here.
Matching CUDA libraries are installed through `tensorflow[and-cuda]`. Use `run.sh`
to expose their library paths for GPU jobs. Other environments are unchanged.

```bash
source ~/.config/cluster/shell.sh
cenv env-speech
python -m pip install -r 00_reference/requirements.txt
python 03_src/pipeline/prepare_wingert.py
```

Select **Python (env-speech)** in the notebook. If an editor cannot find the kernel,
select `/clusterfs/bebb/users/tonychae01/envs/env-speech/bin/python` directly.
Installed package versions are recorded in `00_reference/requirements.lock.txt`.

## What Runs Now

`03_src/01_strf_dstrf.ipynb` runs a short CPU inference/gradient check:
a known linear STRF, one official demo CNN output, finite differences, and centered PCA.
It does not train models or compare unmatched CNN outputs to recorded neurons.

## First Batch Run — Completed

Follow `00_reference/cluster_gpu_workflow.md`: check `gpu_status`, then `gpu_check`,
inspect relevant jobs, and check `sinfo` / partition limits before submitting anything.
The initial restricted query returned `Operation not permitted`. An approved retry
succeeded. Job 7235 subsequently completed on `n0004.catscan0` in 32 seconds.
It used CPU because CUDA libraries were not yet installed; those were added afterward.
Job 7234 had failed before logs were created when using the login-only resolved path.
Always `cd` to the explicit `/clusterfs/...` shared path before submission.

For a future run, check resources and use a new output directory:

```bash
source ~/.config/cluster/shell.sh
cd /clusterfs/bebb/users/tonychae01/projects/26Fall_SpeechSubspace
GPU_COUNT=1 GPU_TIME=00:20:00 gpu_submit env-speech bash 03_src/pipeline/run.sh python 03_src/pipeline/train_pilot.py --epochs 30 --width 8 --output 04_checkpoints/NEW_RUN
```

This fits an LN baseline and an 8-channel CNN to the 20 CLT027c neurons, using
unique stimuli for fitting and averaged repeats of a separate sequence for testing.
It writes models, normalization statistics, learned LN STRFs, test correlations,
and unit-0 dSTRFs to `04_checkpoints/wingert_pilot_seed0/`. It refuses to run on
catscan or without a Slurm allocation, and refuses to overwrite an existing run directory.
Check the returned job ID, `squeue`, and `logs/`; do not submit a duplicate run.
The training path completed end to end for the first CPU pilot.
Inspect its outputs in `03_src/02_wingert_pilot.ipynb`.
`check_gpu.py` separately checks GPU inference and parameter/input gradients;
job 7236 passed these checks on the allocated GPU.
CUDA library setup uses `run.sh`; environment activation alone does not add these paths.

This pilot uses a reduced width, a short fixed fit budget, training-only normalization,
normalized squared-error loss, plain Pearson correlations, and no jackknife ensemble. It is not a full reproduction.
The notebook's centered PCA keeps the mean separately. For predictive subspace models,
include the mean-gradient direction when it lies outside the centered-PC span.

## Sources and Current Checkpoint Issue

- [Wingert et al. (2026)](https://www.nature.com/articles/s41593-026-02216-0)
- [Fixed-version dataset and author demo](https://zenodo.org/records/18331549)
- [Official NEMS source](https://github.com/LBHB/NEMS), pinned in requirements.
- [Separate pretrained NEMS CNN32](https://nemspublic.s3.us-west-2.amazonaws.com/sample_models/CNN-gtgram.fs100.ch32-norm.l1.json)

The downloaded paper `models.zip` contains zero files (directories only).
The separate CNN32 demo has 849 outputs from other recordings; it cannot reproduce
CLT027c neural prediction scores. The downloader retrieves only the smallest paper
recording (5.2 MB) via an HTTP range, rather than the full 4.6 GB recording archive.
Asset checksums are recorded in `00_reference/wingert_assets.json`.
