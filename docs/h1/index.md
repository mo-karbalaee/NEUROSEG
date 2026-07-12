# H1 — Self-Supervised Pretraining for Semi-Supervised Segmentation

This folder is a complete, self-contained technical reference for the **H1**
experiment in NEUROSEG. It is written to be read end-to-end when preparing a
report or presentation: it explains *what* the experiment tests, *how* the model
is built, *how* the two training regimes work, and *how* the comparison is set up
and measured.

Everything here reflects the actual implementation in `src/neuroseg/`, including
the JEPA fixes (EMA target encoder, encoder-shaping reconstruction), the
configurable fine-tuning learning rate, and the memory-bounded (lazy) data
loading.

---

## The hypothesis

> **H1 — Self-supervised pretraining improves semi-supervised segmentation.**
> A JEPA-style model pretrained on *unlabeled* calcium-imaging video learns
> representations that, once a small segmentation head is attached and fine-tuned
> on *labeled* data, outperform an identical model trained from scratch —
> especially when labeled data is scarce.

To test this we train two models of **identical architecture** and compare them
across increasing amounts of labeled data:

| Model | Encoder initialization | What it isolates |
| ----- | ---------------------- | ---------------- |
| **JEPA-pretrained (finetune)** | weights from self-supervised pretraining | the value of pretraining |
| **Supervised baseline** | random weights | the "from scratch" control |

If pretraining helps, the pretrained curve sits **above** the baseline curve, and
the gap is **largest at small labeled fractions**.

---

## The pipeline in one picture

```
                    ┌──────────────────────────────────────────┐
   unlabeled video  │  1. JEPA PRETRAINING (self-supervised)    │
   (whole dataset)  │     encoder + predictor + EMA target      │──► pretrained
                    │     + pixel reconstruction                │    encoder .pt
                    └──────────────────────────────────────────┘        │
                                                                         │ load weights
   labeled video    ┌──────────────────────────────────────────┐        ▼
   at fractions     │  2a. FINE-TUNE (pretrained encoder + head)│──► compound .pt
   {0.1,0.5,        │  2b. SUPERVISED BASELINE (random + head)  │──► compound .pt
    0.75,1.0}       └──────────────────────────────────────────┘        │
                                                                         ▼
                    ┌──────────────────────────────────────────┐
                    │  3. COMPARE Dice / mIoU vs labeled fraction│──► figures + CSV
                    └──────────────────────────────────────────┘
```

---

## Reading order

| # | Document | Covers |
| - | -------- | ------ |
| 1 | [Architecture](01-architecture.md) | Every model component: ResNet5 encoder, ResUNet predictor, VICReg regularizer, EMA target encoder, reconstruction decoder, segmentation head — with tensor shapes |
| 2 | [JEPA pretraining](02-jepa-pretraining.md) | The self-supervised objective, how collapse is prevented, the loss terms, the optimizer, the training loop, the saved checkpoint |
| 3 | [Supervised fine-tuning](03-supervised-finetuning.md) | How the segmentation head is trained, the pretrained-vs-baseline comparison, the learning-rate design |
| 4 | [Data, splits & protocol](04-data-splits-and-protocol.md) | Neurofinder format, lazy loading, clip extraction, labeled fractions, train/val/test splits, the full experiment loop |
| 5 | [Metrics, logging & outputs](05-metrics-logging-and-outputs.md) | Dice, mIoU, the CSV log schema, the figures produced |
| 6 | [Results & fixes](06-results-and-fixes.md) | The earlier negative result, its diagnosis, and the changes made to address it |

---

## Key source files

| File | Role in H1 |
| ---- | ---------- |
| `src/neuroseg/trainers/jepa.py` | All model classes + `build_jepa()`, `build_seg_head()` |
| `src/neuroseg/trainers/h1_trainer.py` | `H1Config`, `pretrain()`, `finetune()`, `run_h1()` |
| `src/neuroseg/trainers/dataset.py` | `NeurofinderDataset` (lazy), format detection |
| `src/neuroseg/metrics.py` | `dice()`, `miou()` |
| `src/neuroseg/checkpoint.py` | checkpoint saving (plain + compound) |
| `src/neuroseg/logger.py` | CSV run logger (`runs.csv`) |
| `src/neuroseg/plots.py` | training-curve and comparison figures |
| `config.yaml` | the hyperparameters used for a real H1 run |

---

## One-line glossary

- **JEPA** — Joint-Embedding Predictive Architecture: predict the *latent
  embedding* of future frames from past frames, rather than the raw pixels.
- **Self-supervised / pretraining** — learning from unlabeled data by solving a
  pretext task (here: predict future latents + reconstruct pixels).
- **Fine-tuning** — continuing to train pretrained weights on the downstream
  labeled task (segmentation).
- **Supervised baseline** — the same network trained only on labels, from random
  init; the control that pretraining must beat.
- **Labeled fraction** — the portion of labeled clips used for fine-tuning
  (1.0 = all, 0.1 = 10 %); the x-axis of the H1 result.
- **EMA target encoder** — a slowly-updated copy of the encoder used as the
  prediction target to prevent representation collapse.
- **VICReg** — variance/covariance regularization that keeps embeddings from
  collapsing to a constant.
- **Dice / mIoU** — segmentation-quality metrics (overlap of predicted vs true
  neuron pixels).
