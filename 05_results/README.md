# Experiment Results

## 2026-11-07 — STRF/dSTRF Method Check

**Status:** Passed the small method check. A small neural-data fit also completed (below); full paper-result reproduction remains pending.

- **Data:** The smallest public Wingert recording, CLT027c: 32 cochleagram channels,
  100 Hz, 20 neurons, approximately 279 s. Preparation produced ten unique 22-s
  training sequences and one separate 22-s test sequence averaged over two repeats.
  Scaling uses training data only. No neural responses were used for fitting yet.
- **Models:** A manually specified linear STRF and the official NEMS demo CNN32,
  output `ARM029a-04-1`. The demo CNN was trained on other recordings and is not
  evaluated against CLT027c neurons.
- **Probe:** 64 contexts within one training sequence, 25 input bins, CPU only.
  No model training or GPU allocation was performed.

| Check | Observed result |
| --- | --- |
| Known linear filter vs. its input gradient | Maximum absolute error: 0 |
| CNN gradient vs. finite differences | Maximum absolute error: 1.47e-6 |
| TensorFlow vs. NEMS forward pass | Maximum absolute error: 1.73e-17 |
| Contexts with nonzero CNN gradients | 41 / 64 |
| Centered PCA dimensions for 95% sampled gradient variance | 3 |

**Interpretation:** The linear model has a fixed filter; this CNN output has
context-dependent sensitivities. Gradient extraction and lag orientation passed
numerical checks. The PCA count describes this small sample only; it is not a
biological dimensionality estimate or a population-level result. The mean CNN
gradient is not a fitted STRF. Centered PCs describe deviations around that mean.

**Limits:** The first restricted Slurm query returned `Operation not permitted`;
an approved retry later succeeded. The downloaded paper `models.zip` contains
folders but zero checkpoint files. A separate demo CNN therefore supports only
method validation. The method-check notebook itself does not test matched-neuron predictions.
SSRF fitting, jackknife/shrinkage, and speech-model analysis remain pending.

**Next at this stage:** Run the reduced-width CLT027c fit; this was subsequently completed below.

Files: [executed notebook](../03_src/01_strf_dstrf.ipynb),
[STRF/dSTRF figure](wingert_smoke/strf_dstrf.png), [PCA figure](wingert_smoke/pca.png),
[metrics](wingert_smoke/metrics.json), and local regenerable `wingert_smoke/filters.npz`.
Sources and run instructions: [pipeline](../03_src/pipeline/README.md).

## 2026-11-07 — First Neural-Data Fit and GPU Setup

**Status:** Job 7235 completed successfully on `n0004.catscan0` in 32 s.

- Same CLT027c split and training-only normalization as above; all 20 neurons.
- Eight-channel shared LN/CNN pilot, seed 0, 30 epochs per model (15 per stage),
  normalized squared-error loss, L2 regularization 1e-4. No hyperparameter selection
  on test data, no noise correction, no jackknife ensemble.
- TensorFlow used **CPU on the compute node**, despite a GPU allocation, because
  CUDA runtime libraries were missing at the time. No training ran on the login node.

| Measurement | Result |
| --- | --- |
| Median held-out Pearson r, LN | 0.039 |
| Median held-out Pearson r, CNN | 0.131 |
| Units with CNN r greater than LN r | 10 / 20 |
| Preselected unit 0, LN / CNN r | 0.148 / 0.172 |
| Unit 0, centered dSTRF PCA dimensions at 95% | 1 |

**Interpretation:** The full small-data path works: fitting, held-out prediction,
learned LN STRFs, and CNN dSTRFs. Prediction quality is modest and unit 0's sampled
dSTRF variation is almost one-dimensional. This short fit does not establish the
paper's findings; underfitting, output gain, limited stimuli, and repeat noise remain
plausible explanations. The test sequence must not guide further model selection.

**GPU setup:** Installed TensorFlow's matching CUDA dependencies after the pilot.
Job 7236 then passed actual GPU inference, parameter-gradient, and input-gradient
checks using `pipeline/run.sh`; it completed in 16 s. Both jobs finished and their
allocations were released. Job 7234 failed before log creation; explicitly using
the shared `/clusterfs/...` submission path resolved that launch issue.

Files: [executed pilot notebook](../03_src/02_wingert_pilot.ipynb),
[predictions](wingert_pilot/predictions.png), [learned filters](wingert_pilot/learned_filters.png),
[metrics](wingert_pilot/metrics.json), [GPU check](wingert_smoke/gpu_check.json).
Models, normalization parameters, and probe arrays are in local
`04_checkpoints/wingert_pilot_seed0/`. Initial CPU-fit metadata remains unchanged.

**Next:** Check fit convergence with training-only validation, then consider a wider
CNN and more gradient contexts. SSRF fitting and shared population analyses follow
after a sufficiently predictive single-site model is established.
