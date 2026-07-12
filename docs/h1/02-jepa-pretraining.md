# H1 · 2. JEPA Pretraining (self-supervised)

Pretraining is where the model learns useful representations **without any
labels**, purely from unlabeled calcium-imaging video. It is implemented in
`pretrain()` in `src/neuroseg/trainers/h1_trainer.py`.

The goal: produce an encoder whose feature maps are (a) informative about neuron
structure and (b) a good warm start for the downstream segmentation head.

---

## 2.1 The pretext task

A **Joint-Embedding Predictive Architecture (JEPA)** does not reconstruct raw
future pixels. Instead it predicts the *latent embedding* of the video, produced
by a target encoder. Two complementary objectives are optimized at once:

1. **Predictive objective (JEPA).** From the encoder's state sequence, roll the
   predictor forward and require the predicted next-states to match the **EMA
   target encoder's** states.
2. **Reconstruction objective.** Decode each latent state back to the original
   grayscale frame (pixel MSE). This grounds the latent space in real spatial
   detail.

On top of the predictive objective sits the **VICReg regularizer**, which keeps
the embeddings from collapsing.

---

## 2.2 The total loss

Per batch, three quantities are computed:

```
L_VC    = std_coeff · std_loss + cov_coeff · cov_loss     (variance + covariance, on online states)
L_pred  = MSE( proj(target_states), proj(predicted_states) )   (predict the EMA target)
L_recon = MSE( decoder(online_states), input_frames )     (pixel reconstruction, trains the encoder)
```

and combined as:

```python
jepa_loss = L_VC + L_pred                 # returned by jepa.unroll(...)
total     = jepa_loss + recon_coeff · L_recon      # recon_coeff = 1.0
total.backward()
```

- `L_VC` prevents dimensional collapse.
- `L_pred` is the actual "predict the future" signal — but computed against a
  **stop-gradient EMA target**, so the encoder cannot cheat by making itself
  trivially predictable.
- `L_recon` forces the encoder to keep pixel-localized information, which is what
  the segmentation head will need.

---

## 2.3 How the predictor is unrolled

`JEPA.unroll(observations, nsteps=steps, unroll_mode="parallel")` does the work.
In `parallel` mode (used here, with `steps=4`):

```python
state  = encoder(observations)              # online states  (B, 8, T, H, W)
target = target_encoder(observations)       # EMA target states, detached
predicted = state
for _ in range(nsteps):
    predicted = predictor(predicted)[:, :, :-1]                 # predict next states
    predicted = cat([state[:, :, :context_length], predicted])  # re-anchor on real context
    L_pred += predcost(target, predicted) / nsteps              # compare to the EMA target
```

The key line is `predcost(target, predicted)`: predictions are scored against the
**target encoder** (detached), not against the online `state`. This is the
stop-gradient that makes the objective non-trivial. `context_length = 2` real
states are kept as anchors on each unroll step.

---

## 2.4 Preventing collapse — the two safeguards

Representation collapse (all embeddings become identical/constant, driving the
prediction loss to zero while learning nothing) is the classic failure mode of
predictive SSL. H1 guards against it two ways:

| Safeguard | Mechanism |
| --------- | --------- |
| **EMA target encoder** | The prediction target comes from a separate encoder that receives **no gradient** and changes only slowly (`ema_momentum = 0.996`). The online encoder cannot collapse the target to meet itself. |
| **VICReg (`L_VC`)** | The variance hinge forces every embedding dimension to keep a minimum spread; the covariance term forces dimensions to be decorrelated. |

Together they keep the latent space high-variance and informative.

---

## 2.5 The optimizer

Two parameter groups:

```python
Adam([
    {"params": jepa.parameters(),                "lr": lr},        # encoder + predictor + projector
    {"params": pixel_decoder.head.parameters(),  "lr": lr / 10},   # reconstruction decoder
])
```

`jepa.parameters()` includes the online encoder and predictor (the target encoder
has `requires_grad = False`, so it contributes no gradients even though it is part
of the module). The decoder head trains at a tenth of the base rate.

After each `optimizer.step()`:

```python
jepa.update_target()      # EMA nudge of target_encoder toward encoder
```

---

## 2.6 The training loop (per epoch)

```
for batch in train_loader:
    x = batch["video"]                                   # (B, 1, T, H, W)
    _, (jepa_loss, ...) = jepa.unroll(x, nsteps=steps)   # L_VC + L_pred
    recon_loss = pixel_decoder(x, x)                     # L_recon
    (jepa_loss + recon_coeff · recon_loss).backward()
    optimizer.step()
    jepa.update_target()
# end of epoch → validate, log to runs.csv
```

**Validation** (`_validate_pretrain`, under `torch.inference_mode()`) reports the
JEPA loss and reconstruction loss on a held-out split (`val_split`, default 0.1).
There are no Dice/mIoU here — pretraining has no labels.

---

## 2.7 Data

Pretraining uses the **unlabeled** view of the data (`NeurofinderDataset(...,
labeled=False)`), which returns only `{"video": ...}`. Because the loader is lazy
and pools every Neurofinder set found under `--data`, you can pretrain on the
**entire dataset** at once. Overlapping clips (`pretrain_clip_stride`, default 2)
multiply the number of training clips. See
[Data, splits & protocol](04-data-splits-and-protocol.md).

---

## 2.8 The saved checkpoint

At the end of pretraining, `save_checkpoint()` writes the **JEPA `state_dict`
only** (encoder + predictor + target encoder + projector), plus a JSON sidecar
recording `hypothesis=H1`, `mode=pretrain`, and the full `arch` dict:

```
jepa_pretrained_h1_<run_id>.pt      # weights
jepa_pretrained_h1_<run_id>.json    # metadata (not marked "compound")
```

This is a **plain** (non-compound) checkpoint: it has no segmentation head, so it
does **not** appear in the inference checkpoint picker. Its only consumer is the
fine-tuning phase, which loads the encoder weights out of it.

A training-curve figure (`pretrain_..._curves.png`: JEPA loss and reconstruction
loss over epochs) is generated automatically.

Next: how those pretrained weights are turned into a segmentation model, and how
the comparison against the from-scratch baseline is set up —
[Supervised fine-tuning](03-supervised-finetuning.md).
