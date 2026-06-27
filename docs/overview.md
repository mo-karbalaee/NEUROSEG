# NEUROSEG — Scientific Overview

## What This Project Does

NEUROSEG is a neural segmentation pipeline for **calcium imaging data**. Given a time-lapse TIFF recording of brain tissue, it locates individual neuron cell bodies (somas), draws segmentation masks around each one, and extracts their fluorescence activity over time as ΔF/F₀ traces.

The system has two operating modes:

- **Inference** — Load a TIFF stack, segment neurons, extract activity traces, save visualisations.
- **Training** — Train a JEPA-style self-supervised model on calcium imaging data and evaluate three experimental hypotheses about representation learning.

---

## The Scientific Problem

### Calcium Imaging

Calcium imaging is a technique in systems neuroscience where neurons are genetically modified to express a fluorescent protein (GCaMP) that brightens when calcium ions flood the cell — which happens when the neuron fires. A two-photon microscope records the tissue as a time-series of 2-D images (a TIFF stack). Each frame shows the fluorescence intensity across a field of view.

The result is a video where individual bright blobs pulse as neurons fire. The goal is to:
1. Find every neuron soma in the field of view — **segmentation**.
2. Extract its fluorescence trace over time — **signal extraction**.

### Why Segmentation Is Hard

Calcium imaging presents several challenges that make standard computer vision approaches fail or degrade:

- **Low SNR**: Fluorescence signals are weak; background scattering and noise can be comparable in intensity to a silent neuron.
- **Overlap**: Dendrites, neuropil, and somas pile on top of each other in 2-D projections from 3-D tissue.
- **Temporal dynamics**: Neuron brightness changes with firing — a neuron that is silent for 1000 frames can look indistinguishable from background, then suddenly brighten.
- **Scarcity of labels**: Manually annotating neuron locations in thousands of frames across many sessions is impractical. Most large-scale recordings have no pixel-level annotations.

### Self-Supervised Learning as a Solution

The core idea in NEUROSEG is that a model trained to **predict its own future representations** — without any labels — will learn to encode the temporal structure of neural activity. When subsequently fine-tuned on a small number of labeled frames, those learned representations should give it a head start that a randomly-initialised model cannot match.

This is the Joint Embedding Predictive Architecture (JEPA) paradigm applied to neuroscience imaging data.

---

## The Three Hypotheses

NEUROSEG formalises three research questions as testable experimental hypotheses:

### H1 — Self-supervised pretraining improves semi-supervised segmentation

If the JEPA model learns meaningful structure from unlabeled calcium imaging data, then fine-tuning it with a small fraction of labeled data should outperform a supervised baseline trained from scratch on the same fraction. The improvement should be largest when labels are most scarce.

**What it tests:** Whether learned temporal representations transfer to the segmentation task under label-scarce conditions.

### H2 — Learned representations generalise across organisms

Calcium imaging is performed across species — mouse, zebrafish, Drosophila, and others. Labeled annotations are far more abundant for some organisms (mouse cortex) than others. If JEPA representations capture general features of neural activity rather than organism-specific appearance, a model pretrained on one organism should transfer to another with minimal fine-tuning, losing less performance than a supervised model trained and transferred under identical conditions.

**What it tests:** Cross-organism generalisability of self-supervised representations vs. supervised ones.

### H3 — Learned representations are more temporally stable per neuron

A good neural representation should encode identity: the same neuron at different time points should map to nearby points in embedding space, while different neurons should be separated. This property — temporal stability — is not enforced by any explicit training signal. H3 asks whether JEPA pretraining induces it as an emergent property of learning to predict future states.

**What it tests:** Whether the within-neuron cosine similarity gap (within − between) is larger for JEPA-pretrained encoders than for supervised or randomly-initialised ones.

---

## Dataset: Neurofinder

NEUROSEG operates on the **Neurofinder** benchmark dataset — the standard benchmark for calcium imaging neuron segmentation. It provides:

- Multi-frame TIFF stacks of calcium imaging recordings from multiple species and brain regions.
- Pixel-level annotations in `regions/regions.json` (list of pixel coordinates per neuron).
- Standardised directory structure (`images/image*.tiff` + `regions/regions.json`).

| Dataset prefix | Organism | Brain region |
| -------------- | -------- | ------------ |
| `neurofinder.00`–`03` | Mouse | Visual cortex |
| `neurofinder.04` | Zebrafish | Whole brain |
| `neurofinder.10` | Mouse | Hippocampus |

The organism label is inferred automatically from the directory name, enabling organism-agnostic experiment configuration.

---

## Output of a Complete Run

After a training + inference run, the project produces:

| Output | Description |
| ------ | ----------- |
| `output/checkpoints/*.pt` | Trained JEPA model weights |
| `output/segmentation/<file>/frame_N.png` | Segmentation overlays per frame |
| `output/traces/<file>/traces_combined.png` | All neuron ΔF/F₀ traces on one plot |
| `output/traces/<file>/neuron_N.png` | Individual trace per neuron |
| `output/traces/traces+<file>.npy` | Raw ΔF/F₀ array (N neurons × T frames) |
| `mlruns/` | MLflow experiment store (metrics, params, artifacts) |
| `output/figures/h1_dice_comparison.png` | H1 result figure: Dice vs labeled fraction |
| `output/figures/segmentation_preview.png` | Segmentation preview for demo/paper |

---

## Related Work

NEUROSEG draws on the following bodies of work:

- **JEPA / I-JEPA**: Assran et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR 2023. The core self-supervised training objective.
- **VICReg**: Bardes et al. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR 2022. The regulariser (VCLoss) preventing representation collapse.
- **Neurofinder**: Berens et al. (2018). "Community-based benchmarking improves spike rate inference from two-photon calcium imaging data." PLOS Computational Biology. The benchmark used for training and evaluation.
- **Cellpose**: Stringer et al. (2021). "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods. Used as the supervised inference fallback when no JEPA checkpoint is selected.
