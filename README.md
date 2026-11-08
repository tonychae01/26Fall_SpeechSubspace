# speech-subspace-alignment
Analyzing Population Subspace Structure in Pretrained Speech Models and Alignment with Auditory Cortex Circuits

**Beyond Single-Unit STRFs: Understanding Auditory Tuning Subspaces in HuBERT and wav2vec 2.0**
Tony Chae, Stefan Hajduk

## Repository Structure

```text
00_reference/       Project plan, progress log, and requirements.txt
01_raw_data/        Raw data
02_processed_data/  Processed data
03_src/             Experiment and analysis code
04_checkpoints/     Model and training checkpoints
05_results/         Experiment summaries and figures
06_external/        External code and resources
```

- [Project plan](00_reference/project_plan.md): Current working draft, subject to revision.
- [Progress log](00_reference/progress.md): Completion status, current stage, and next steps.
- [Results log](05_results/README.md): Concise summaries of experiments that have actually been run.
- [Cluster GPU workflow](00_reference/cluster_gpu_workflow.md): Shell setup, allocation checks, and batch / interactive job guidance.

## First Notebook and Environment

Start with [STRF and dSTRF basics](03_src/01_strf_dstrf.ipynb). It checks a known linear filter,
probes one official NEMS demo CNN output, and extracts a small tuning subspace.
English explanations and outputs are included in the notebook.

The environment is `/clusterfs/bebb/users/tonychae01/envs/env-speech` (Python 3.10).
Select **Python (env-speech)** in the notebook, or activate it explicitly:

```bash
source ~/.config/cluster/shell.sh
cenv env-speech
```

See [pipeline instructions](03_src/pipeline/README.md) for dependencies, data preparation,
and repeatable LN/CNN batch fits. The first small neural-data fit completed; inspect
[its results notebook](03_src/02_wingert_pilot.ipynb). This is an initial method pilot,
not a full paper reproduction. HuBERT and wav2vec 2.0 experiments have not started.
