
# H1 · 1. Architecture

Everything in H1 is built from a small set of components defined in
`src/neuroseg/trainers/jepa.py` and assembled by two factory functions:

- `build_jepa(arch, device)` → the full self-supervised model (`JEPA`)
- `build_seg_head(dstc, hidden)` → the tiny segmentation head used for fine-tuning

This page walks through each component and the tensor shapes that flow through
them. Shapes use the reference config (`config.yaml`): input channels `dobs=1`,
encoder width `henc=32`, predictor width `hpre=32`, **latent channels `dstc=8`**,
image size `128×128`, clip length `seq_len=5`, batch size `B`.

---

## 1.1 The big picture

The `JEPA` object bundles five things:

```python
JEPA(encoder, action_encoder, predictor, regularizer, predcost,
     target_encoder=..., ema_momentum=0.996)
```

| Component | Class | Purpose |
| --------- | ----- | ------- |
| `encoder` | `ResNet5` | maps a video clip → a sequence of latent feature maps ("states") |
| `target_encoder` | `ResNet5` (EMA copy) | produces the **prediction targets**; not trained by gradient |
| `predictor` | `StateOnlyPredictor` → `ResUNet` | predicts the next latent state from consecutive states |
| `regularizer` | `VCLoss` | VICReg variance+covariance loss that prevents collapse |
| `predcost` | `SquareLossSeq` | MSE between predicted and target latent states |
| `action_encoder` | (unused) | placeholder — calcium imaging has no actions |

Separately, two heads are attached depending on the phase:

- **Pretraining** adds an `ImageDecoder` (pixel reconstruction).
- **Fine-tuning** adds a `build_seg_head(...)` (per-pixel segmentation).

---

## 1.2 The encoder — `ResNet5`

A 5-layer residual CNN that turns each frame into a `dstc`-channel feature map.

```
Conv2d(1 → 32, 3×3) + BN + ReLU
ResidualBlock(32 → 32,  stride=1)
ResidualBlock(32 → 64,  stride=1)
ResidualBlock(64 → 8,   stride=1)      # 8 = dstc
```

**Critical design choice: all strides = 1.** The encoder never downsamples, so the
output feature map has the **same spatial size as the input** (128×128). This is
required so the segmentation head can produce a spatially-aligned mask without
learned upsampling. (Do not add downsampling strides here — it would misalign the
mask.)

The encoder is wrapped in `TemporalBatchMixin`, which lets a 2-D CNN process a
5-D video tensor: it folds time into the batch dimension, runs the 2-D forward,
then unfolds:

```
(B, C, T, H, W)  --reshape-->  (B·T, C, H, W)  --ResNet5-->  (B·T, dstc, H, W)  --reshape-->  (B, dstc, T, H, W)
```

**Shape:**

```
input   (B, 1, 5, 128, 128)   grayscale video clip
output  (B, 8, 5, 128, 128)   latent "state" sequence — spatial dims preserved
```

---

## 1.3 The predictor — `StateOnlyPredictor` wrapping `ResUNet`

The predictor answers: *given the latent states, what comes next?* It takes each
pair of consecutive states, concatenates them along the channel dimension, and
predicts the next-state feature map.

```python
prev = state[:, :, :-1]                 # (B, 8, T-1, H, W)
next = state[:, :, 1:]                  # (B, 8, T-1, H, W)
combined = cat([prev, next], dim=1)     # (B, 16, T-1, H, W)
predicted = ResUNet(combined)           # (B, 8, T-1, H, W)
```

The `ResUNet` is a standard U-Net (encoder path with stride-2 downsampling,
bottleneck, decoder path with transposed-conv upsampling and skip connections),
also wrapped in `TemporalBatchMixin`. Unlike the `ResNet5` encoder, the U-Net
*does* downsample internally (that is fine — it upsamples back before output).

```
input   (B, 16, T-1, 128, 128)
  enc1  (·, 32,  128, 128)
  enc2  (·, 64,  64,  64)
  enc3  (·, 128, 32,  32)
  bott  (·, 256, 16,  16)
  dec3  (·, 128, 32,  32)
  dec2  (·, 64,  64,  64)
  dec1  (·, 32,  128, 128)
  head  (·, 8,   128, 128)
output  (B, 8, T-1, 128, 128)   predicted next-states
```

`context_length` (default 2) controls how many leading real states are kept as
context when the predictor is unrolled for multiple steps (see
[JEPA pretraining](02-jepa-pretraining.md)).

---

## 1.4 The EMA target encoder

A **second `ResNet5`, identical in shape to the encoder**, created inside
`build_jepa`:

```python
target_encoder = ResNet5(dobs, henc, dstc)
target_encoder.load_state_dict(encoder.state_dict())   # starts equal
# its parameters have requires_grad = False
```

The predictor's job is to predict the target encoder's embeddings of the video,
**not** the online encoder's own output. The target encoder is never updated by
gradient descent; instead, after every optimizer step it is nudged toward the
online encoder with an exponential moving average:

```python
target ← ema_momentum · target + (1 − ema_momentum) · online      # ema_momentum = 0.996
```

**Why this exists:** without a separate, stop-gradient target, a JEPA can cheat —
the encoder can collapse to trivially predictable (e.g. constant) features that
make the prediction loss zero while carrying no useful information. A slowly
moving target that receives no gradient removes that shortcut. This is the single
most important anti-collapse mechanism, and it was added specifically to fix a
prior failure (see [Results & fixes](06-results-and-fixes.md)).

---

## 1.5 The regularizer — `VCLoss` (VICReg)

A second line of defense against collapse, applied to the **online** encoder's
states. It has two terms:

- **Variance (hinge) loss** — penalizes any embedding dimension whose standard
  deviation across the batch drops below a margin (`std_margin = 1.0`). This
  forces each dimension to actually vary.
  ```
  std_loss = mean( relu( std_margin − std(features) ) )
  ```
- **Covariance loss** — penalizes correlation between different embedding
  dimensions (off-diagonal covariance), pushing the dimensions to encode
  different information.
  ```
  cov_loss = mean( off_diagonal( cov(features) )² )
  ```

Combined with coefficients from the config:

```
L_VC = std_coeff · std_loss + cov_coeff · cov_loss      # std_coeff=10, cov_coeff=100
```

Features are first passed through a small MLP `Projector` (`dstc → 4·dstc →
4·dstc`) before the variance/covariance statistics are computed — a standard
VICReg detail.

---

## 1.6 The predictive cost — `SquareLossSeq`

Simply the mean-squared error between the predicted states and the (projected)
target states:

```
L_pred = MSE( proj(target_states), proj(predicted_states) )
```

The same `Projector` used by the regularizer is reused here.

---

## 1.7 The reconstruction decoder — `ImageDecoder` + `JEPAProbe`

A small 2-layer CNN that maps a latent feature map back to a grayscale image:

```
Conv2d(8 → 16, 3×3) + ReLU + Conv2d(16 → 1, 3×3)
```

It is wrapped by `JEPAProbe`, which runs the encoder, feeds the state to the
decoder, and computes MSE against the original frames:

```python
JEPAProbe(jepa, ImageDecoder(...), MSELoss(), train_encoder=True)
```

With `train_encoder=True` (used in pretraining), the reconstruction gradient
**flows into the encoder**, pushing it to retain spatially-localized pixel detail
— which is exactly what a dense segmentation head needs downstream. (Previously
the encoder was detached here, so reconstruction shaped only the decoder; that was
changed — see [Results & fixes](06-results-and-fixes.md).)

---

## 1.8 The segmentation head — `build_seg_head`

Attached only during fine-tuning. A tiny per-pixel classifier:

```
Conv2d(8 → 16, 3×3) + ReLU + Conv2d(16 → 1, 1×1) + Sigmoid
```

```
input   (B, 8, 128, 128)     one feature map (temporal mean of the encoder states)
output  (B, 1, 128, 128)     per-pixel neuron probability in [0, 1]
```

The output is bilinearly interpolated up to the ground-truth mask resolution and
thresholded at 0.5 to produce a binary mask.

---

## 1.9 End-to-end shape summary (reference config)

| Stage | Tensor | Shape |
| ----- | ------ | ----- |
| Input clip | `x` | `(B, 1, 5, 128, 128)` |
| Encoder states | `enc(x)` | `(B, 8, 5, 128, 128)` |
| Target states (EMA) | `target_enc(x)` | `(B, 8, 5, 128, 128)` |
| Predicted states | predictor output | `(B, 8, 4, 128, 128)` |
| Temporal mean (fine-tune) | `enc.mean(dim=2)` | `(B, 8, 128, 128)` |
| Seg-head output | `seg_head(mean)` | `(B, 1, 128, 128)` |
| Reconstruction | `decoder(enc)` | `(B, 1, 5, 128, 128)` |

Next: how these components are trained without labels — [JEPA pretraining](02-jepa-pretraining.md).
