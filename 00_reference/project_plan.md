# Project Plan — Working Draft

**Beyond Single-Unit STRFs: Understanding Auditory Tuning Subspaces in HuBERT and wav2vec 2.0**
Tony Chae, Stefan Hajduk
Draft recorded: 2026-11-07

This document is a working plan based on the current proposal and may be revised. The items below are planned work, not experimental results or validated conclusions. Actual progress is tracked in `progress.md`.

## Motivation & Research Question

Conventional STRFs describe units using fixed linear filters and can miss nonlinear, context-dependent tuning. Single-unit attribution or stimulus optimization alone also leaves open how populations collectively encode auditory and speech information.

Our starting point is Wingert et al. (2026), who analyzed context-dependent filters (dSTRFs) from CNN models fitted to ferret auditory cortical neurons. In the proposed interpretation, neurons share low-dimensional sensory subspaces while differing in their nonlinear tuning. We will investigate whether related population organization emerges in pretrained speech models.

**Central question:** Do self-supervised speech models develop low-dimensional, context-dependent auditory tuning subspaces, and does their population organization resemble that of biological auditory cortex?

## Scope & Setup

- Models: pretrained **HuBERT-base** and **wav2vec 2.0-base** only.
- Dataset: **LibriSpeech** speech; natural sounds are an optional extension.
- BirdAVES and birdsong are excluded from the current scope.
- STRFs serve as a baseline; context-dependent and population-level analyses are the main focus.
- Begin with representative early, middle, and late layers and selected units, using a common spectrotemporal input space.
- Exact checkpoints, data split, unit selection, context windows, and the differentiable mapping between spectrograms and waveform-model inputs remain to be specified before extraction.

## Planned Experiments

### 1. Single-Unit Interpretation: STRF vs. dSTRF

Compare fixed STRFs with context-dependent input Jacobians (dSTRFs) across selected units and layers.

**Question:** How does individual-unit acoustic tuning change with stimulus context and network depth?

### 2. Low-Dimensional Subspace Extraction and SSRF Flattening

Apply PCA to time-varying dSTRFs to extract principal tuning filters, initially considering 90–95% explained variance. Fit compact nonlinear SSRFs using these filters and compare held-out prediction accuracy with conventional STRFs. The precise SSRF formulation and fitting protocol will be settled after checking the reference method.

**Question:** How many dimensions explain each unit's nonlinear computation, and how does this vary across layers?

### 3. Layer-Wise Population Subspace Organization

Estimate shared population tuning subspaces from unit dSTRFs. Measure within-layer similarity and cross-layer principal angles / subspace similarity (SSI; exact definition to be confirmed). Compare dimensionality, organization, and acoustic tuning across layers and models.

**Question:** How do shared tuning subspaces emerge and reorganize across network depth and between HuBERT and wav2vec 2.0?

### 4. Comparison with Biological Auditory Cortex

Compare subspace dimensionality, shared tuning structure, and nonlinear selectivity with published A1/PEG results from Wingert et al. (2026). Account for differences in stimuli, input representations, and measurement methods when interpreting similarities.

**Question:** Which organizational principles of biological auditory encoding also emerge in self-supervised speech models?

## Optional Extensions

1. Same-architecture random-weight controls to separate architectural effects from learned structure.
2. Spectral and temporal modulation tuning of principal filters; layer-wise linear phonetic probes if feasible.
3. Compare STRFs/dSTRFs with top natural clips or stimuli optimized to drive individual units.
4. If dSTRFs are uninformative, consider model metamers (Feather et al., 2023): synthesized sounds matching a natural sound's layer activity to study information that the layer ignores.
5. Extend the stimulus set to natural sounds if time allows.

## Expected Outcomes

Determine whether low-dimensional nonlinear subspaces explain model computations better than fixed STRFs, how these structures evolve across layers, and whether they show organizational similarities to auditory cortical populations. These are hypotheses to test, not assumed outcomes.

## Tentative Four-Week Plan

| Week | Planned work |
| --- | --- |
| 1 | Model setup, stimulus preparation, and dSTRF extraction |
| 2 | Single-unit comparison, PCA, and SSRF fitting |
| 3 | Population subspace and cross-layer analyses |
| 4 | Cortical comparison, visualization, and final report |

This is a tentative sequence; dates and scope can change. Environment setup is deferred and will use an external `envs` directory.

## References Provided in the Proposal

Bibliographic details below are carried over from the proposal and have not yet been independently checked.

1. Wingert, Parida, Norman-Haignere, David. [Convolutional neural network models describe the encoding subspace of local circuits in auditory cortex](https://www.nature.com/articles/s41593-026-02216-0). Nature Neuroscience 29, 876–887 (2026).
2. Hsu et al. [HuBERT: Self-supervised speech representation learning by masked prediction of hidden units](https://arxiv.org/abs/2106.07447).
3. Baevski et al. [wav2vec 2.0: A framework for self-supervised learning of speech representations](https://arxiv.org/abs/2006.11477).
4. Pasad, Shi, Livescu. [Comparative layer-wise analysis of self-supervised speech models](https://arxiv.org/abs/2211.03929). ICASSP (2023).
5. Huo, Dunbar. [Iterative refinement, not training objective, makes HuBERT behave differently from wav2vec 2.0](https://arxiv.org/abs/2508.08110). Preprint (2025).
6. Meng, Goldwater, Tang. [Effective context in neural speech models](https://www.isca-archive.org/interspeech_2025/meng25b_interspeech.html). Interspeech (2025).

Dataset: [LibriSpeech](https://www.openslr.org/12). Optional stimuli: [Natural sounds](https://mcdermottlab.mit.edu/svnh/Natural-Sound/Stimuli.html).
