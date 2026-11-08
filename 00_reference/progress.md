# Progress Log

Last updated: 2026-11-07

## Current Status

The external `env-speech` environment is installed, including matching TensorFlow
CUDA libraries. Two short English notebooks have executed: STRF/dSTRF numerical
checks and a first small LN/CNN fit on real CLT027c neurons. Job 7235 completed the
fit on a compute-node CPU; after CUDA setup, job 7236 passed GPU inference and
gradient checks. These are initial pilots, not a full paper-result reproduction.

## Checklist

- [x] Create the project folder structure and working plan
- [x] Keep project documentation in English
- [x] Install an isolated Python 3.10 environment in external `envs/env-speech`
- [x] Record requirements and installed versions; check package consistency
- [x] Download one small paper recording and inspect official example code
- [x] Execute a known-STRF / demo-CNN dSTRF / centered-PCA notebook
- [x] Prepare a separate batch pipeline for the first small LN/CNN fit
- [x] Run and evaluate the first small fit on actual CLT027c neurons
- [x] Validate actual GPU inference and input / parameter gradients
- [ ] Reproduce paper prediction / subspace findings with appropriate controls
- [ ] Prepare HuBERT, wav2vec 2.0, and LibriSpeech
- [ ] Exp. 1: Speech-model STRF / dSTRF comparison
- [ ] Exp. 2: PCA and SSRF fitting / held-out evaluation
- [ ] Exp. 3: Layer-wise population subspaces
- [ ] Exp. 4: Compare with published auditory cortex results
- [ ] Final figures and report

## Next Step and Limitations

Assess convergence with training-only validation before expanding width or scope.
The first short pilot has modest prediction quality (median test r: LN 0.039,
CNN 0.131). Do not use its final test sequence for model selection.

The initial restricted Slurm query failed; an approved retry succeeded. Jobs 7235
and 7236 both completed with exit 0. Job 7234 failed before log creation; explicit
submission from the shared `/clusterfs/...` path resolved that launch issue.

The paper model ZIP has directories but no checkpoint files. The first notebook
uses a separate pretrained NEMS demo CNN; the second uses newly fitted small
models on actual CLT027c data. Neither claims full paper reproduction.

See `../05_results/README.md` for observed results and `../03_src/pipeline/README.md`
for commands and sources. Keep advancing one checked step at a time.

## Change Log

- 2026-11-07: Prepared the structure, plan, progress log, and results log; translated documentation into English.
- 2026-11-07: Added cluster workflow, installed env-speech, prepared CLT027c, executed both notebooks, completed the small neural-data fit, and verified GPU support. Full paper reproduction remains pending.
