# H1 · 3. Supervised Fine-Tuning and the Baseline

This is the supervised half of H1: attach a segmentation head, train on labeled
data, and measure Dice/mIoU. The **same function** produces both arms of the
comparison — it is implemented in `finetune()` in
`src/neuroseg/trainers/h1_trainer.py`, called once per arm.

---

## 3.1 The two arms — identical architecture, one difference

```python
finetune(pretrained_ckpt, ..., mode="finetune")            # arm A: pretrained init
finetune(None,            ..., mode="supervised_baseline") # arm B: random init
```

Both arms build the **exact same network**:

```python
jepa     = build_jepa(cfg.arch_dict(), device)       # same ResNet5 encoder
seg_head = build_seg_head(cfg.dstc, cfg.seg_head_hidden)   # same head
```

The **only** architectural difference between the two arms is the encoder's
starting weights:

| Arm | `mode` | Encoder init |
| --- | ------ | ------------ |
| **Pretrained (finetune)** | `"finetune"` | loads the pretrained JEPA `state_dict` |
| **Supervised baseline** | `"supervised_baseline"` | random initialization |

This is what makes H1 a clean ablation: same encoder, same head, same data, same
schedule — pretraining is the single variable.

> **The "supervised method used"** in H1 *is* this: the JEPA `ResNet5` encoder
> followed by the 2-conv sigmoid segmentation head, trained end-to-end with binary
> cross-entropy. The baseline is not a different architecture — it is this same
> network trained from scratch.

---

## 3.2 The forward pass

The encoder produces a state per frame; these are **averaged over time** into a
single feature map, which the head turns into a probability mask:

```python
enc_state = jepa.encoder(x)          # (B, 8, T, H, W)
enc_mean  = enc_state.mean(dim=2)    # (B, 8, H, W)      temporal average
pred      = seg_head(enc_mean)       # (B, 1, H, W)      per-pixel probability
pred      = F.interpolate(pred, size=mask.shape[-2:])   # up to mask resolution
loss      = BCELoss(pred, mask[:, 0:1])                 # binary cross-entropy
```

- **Temporal averaging** collapses the `T` frames into one summary map. Neurons
  are roughly stationary within a clip, so a single spatial map is the natural
  target.
- The ground-truth mask is **binary** (neuron vs background) and broadcast across
  frames, so only the first channel is needed.
- Loss is **binary cross-entropy** — note this is a *different loss regime* from
  pretraining (no JEPA/VICReg loss is involved in fine-tuning).

---

## 3.3 The learning-rate design (and why it matters)

Two parameter groups, with the encoder and head trained at different rates:

```python
encoder_lr = lr · finetune_encoder_lr_scale   if mode == "finetune"
             lr                                if mode == "supervised_baseline"

Adam([
    {"params": jepa.encoder.parameters(), "lr": encoder_lr},
    {"params": seg_head.parameters(),     "lr": lr},
])
```

- `finetune_encoder_lr_scale` (config, default `0.1`) sets how fast the
  **pretrained** encoder is allowed to move. A small value is the classic
  fine-tuning recipe — preserve pretrained features, adapt slowly.
- The **baseline** always uses the full `lr` on its encoder (it has nothing to
  preserve — it must learn from scratch).

**Fairness note.** If `finetune_encoder_lr_scale < 1`, the pretrained arm updates
its encoder more slowly than the baseline, which is only fair *if* the pretrained
features are already good. Setting `finetune_encoder_lr_scale = 1.0`
(as in `config.yaml`) makes the two arms differ **only by initialization** — the
cleanest possible H1 comparison, and the recommended setting when the question is
strictly "does pretraining help?". See [Results & fixes](06-results-and-fixes.md)
for why this was made configurable.

Note the encoder is **not frozen** in either arm — this is real fine-tuning, not
linear probing.

---

## 3.4 Validation and test evaluation

- After each epoch, `_validate_finetune()` computes **Dice** and **mIoU** on the
  validation split (threshold 0.5 on the probability map).
- After the final epoch, the same function is run on a **held-out test split** —
  these test numbers are the ones reported in the H1 figures.

See [Metrics, logging & outputs](05-metrics-logging-and-outputs.md) for the exact
metric definitions and the train/val/test split logic (the split logic lives in
[Data, splits & protocol](04-data-splits-and-protocol.md)).

---

## 3.5 The saved checkpoint

Each fine-tune / baseline run saves a **compound checkpoint** via
`save_compound_checkpoint()`:

```
jepa_h1_finetune_f50_<run_id>.pt        # {jepa, seg_head, arch} bundled
jepa_h1_finetune_f50_<run_id>.json      # sidecar: "compound": true, dice, miou, ...
```

Compound checkpoints contain everything needed to run inference (encoder + head +
architecture), so **these are the checkpoints that appear in the inference picker**
and can segment new TIFF stacks. The model name encodes the arm and labeled
fraction (`finetune` vs `supervised`, `f10`/`f50`/`f75`/`f100`).

A per-run training-curve figure (`finetune_..._curves.png`: train loss, val Dice,
val mIoU, final test scores) is generated automatically.

Next: the data, the splits, the labeled fractions, and how the whole experiment is
orchestrated — [Data, splits & protocol](04-data-splits-and-protocol.md).
