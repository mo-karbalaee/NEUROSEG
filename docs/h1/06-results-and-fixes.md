# H1 · 6. Results & Fixes

This page documents the first real H1 result, why it failed, and the concrete
changes made to address it. It is the most useful section for a report: it shows
the scientific reasoning, not just the final code.

---

## 6.1 The first result was a negative result

An initial H1 run (on Kaggle) produced the **opposite** of the hypothesis — the
JEPA-pretrained model lost to the supervised baseline at **every** labeled
fraction:

| Fraction | Pretrained Dice | Baseline Dice | Pretrained mIoU | Baseline mIoU |
| -------- | --------------- | ------------- | --------------- | ------------- |
| 10 % | 0.24 | **0.29** | 0.49 | **0.51** |
| 50 % | 0.59 | **0.82** | 0.65 | **0.82** |
| 75 % | 0.59 | **0.81** | 0.65 | **0.83** |
| 100 % | 0.64 | **0.82** | 0.68 | **0.83** |

Two diagnostic signatures stood out:

1. **Pretraining loss collapsed toward ~0**, yet downstream Dice was poor — the
   hallmark of a *degenerate* self-supervised solution (features that are easy to
   predict but carry no useful information).
2. **The pretrained fine-tune was still improving at 100 epochs** (its validation
   Dice climbed slowly and never plateaued), while the baseline converged by
   ~epoch 20 — a sign the pretrained encoder was being updated too slowly and/or
   starting from unhelpful features.

---

## 6.2 Root causes

| # | Problem | Why it hurt |
| - | ------- | ----------- |
| 1 | **No stop-gradient target.** The predictor was trained to predict the *same* online encoder's output, with no EMA/target network. | The encoder could minimize the prediction loss by collapsing to trivially predictable features — driving loss to ~0 while learning nothing segmentation-relevant. |
| 2 | **Reconstruction never trained the encoder.** The pixel decoder wrapped the encoder in `torch.no_grad()` + `.detach()`. | The only signal shaping the encoder was the (collapse-prone) prediction loss; the pixel-grounding signal was wasted on the decoder alone. |
| 3 | **Hardcoded low fine-tune encoder LR.** The pretrained encoder was fixed at `lr/10`; the baseline used full `lr`. | If pretrained features are bad, a low LR means the encoder can't recover — and the comparison is confounded (the two arms differed in LR as well as init). |
| 4 | **No data advantage.** Pretraining and fine-tuning used the *same single recording*. | H1 requires unlabeled ≫ labeled; with equal data the pretrained model has no extra information to exploit, so no benefit is even possible. |
| 5 | **Out-of-memory on the full dataset.** The dataset loaded every frame of every recording into RAM. | Prevented running on the whole (20 GB) benchmark, which is exactly the large unlabeled pool H1 needs. |

---

## 6.3 The fixes

Each root cause maps to a concrete change (all now in the code described in this
folder):

| # | Fix | Where |
| - | --- | ----- |
| 1 | **EMA target encoder + stop-gradient.** A second, non-trainable `ResNet5` produces the prediction targets, updated by EMA (`ema_momentum = 0.996`). The predictor predicts the detached target, not the online state. | `build_jepa`, `JEPA.unroll`, `JEPA.update_target` — see [Architecture §1.4](01-architecture.md) and [Pretraining §2.3–2.4](02-jepa-pretraining.md) |
| 2 | **Reconstruction trains the encoder.** `JEPAProbe(train_encoder=True)` lets the pixel-MSE gradient flow into the encoder, weighted by `recon_coeff`. | `JEPAProbe`, `pretrain()` — see [Architecture §1.7](01-architecture.md) |
| 3 | **Configurable fine-tune encoder LR.** `finetune_encoder_lr_scale` replaces the hardcoded `lr/10`; set to `1.0` for a same-LR, init-only comparison. | `finetune()` — see [Fine-tuning §3.3](03-supervised-finetuning.md) |
| 4 | **Separate, larger unlabeled pool.** `run_h1` already pretrains on `--data` and fine-tunes on `--labeled-data`; recursive dataset discovery lets `--data` point at the whole benchmark. | `run_h1`, `find_neurofinder_dirs` — see [Protocol §4.6](04-data-splits-and-protocol.md) |
| 5 | **Lazy, memory-bounded loading.** `NeurofinderDataset` indexes clips and reads frames on demand. | `NeurofinderDataset` — see [Protocol §4.3](04-data-splits-and-protocol.md) |

These are also recorded, with symptoms, in the project's engineering notes.

---

## 6.4 What to expect / how to judge the next run

With the fixes, the questions to ask of a new H1 run are:

1. **Does the pretraining loss stay non-trivial?** With the EMA target, the
   prediction loss should *not* crash to ~0 immediately. A healthy run shows the
   VICReg terms keeping variance up.
2. **Is there signal at low fractions?** The clearest evidence for H1 is at the
   **0.1** fraction: the baseline has too few labels to learn a good encoder, so
   pretraining should show its largest advantage there.
3. **Does the pretrained arm converge, not just crawl?** With
   `finetune_encoder_lr_scale = 1.0`, the pretrained arm should converge on a
   similar timescale to the baseline — if it still lags, the features are still
   the limiting factor.

---

## 6.5 If it still under-performs — next levers

Ordered by expected value:

1. **Linear-probe diagnostic.** Freeze the pretrained encoder and train only the
   seg head. This measures representation quality *directly* in minutes, isolating
   "bad features" from "bad fine-tuning schedule." (Not yet implemented — the
   recommended next diagnostic.)
2. **More / more diverse unlabeled data.** SSL scales with data; a larger, more
   varied unlabeled pool is the most reliable lever once the objective is sound.
3. **Stronger spatial pretext.** If temporal prediction alone under-delivers for
   dense segmentation, increase the weight of the reconstruction term
   (`recon_coeff`) or add a masked-reconstruction (MAE-style) objective, which
   tends to produce features better suited to per-pixel tasks.
4. **Tune the EMA momentum / VICReg coefficients** if collapse or instability
   reappears.

---

## 6.6 Summary

H1's current design is the product of a debugging cycle: a first run showed
pretraining *hurting*, which was traced to a collapse-prone objective, a wasted
reconstruction signal, a confounded learning-rate comparison, and an
unfairly-small unlabeled pool. The architecture and protocol documented in this
folder are the corrected version. Whether pretraining now *helps* is the empirical
question the next run answers — and the figures in
[Metrics, logging & outputs §5.5](05-metrics-logging-and-outputs.md) are how you
read that answer.
