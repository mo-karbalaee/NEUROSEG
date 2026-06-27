# Model Architecture

## Design Philosophy

NEUROSEG uses a **Joint Embedding Predictive Architecture (JEPA)**. The central idea is that the model never tries to reconstruct pixels. Instead, it learns to predict the future **representation** of a frame from the present representation. Representations live in a learned latent space, not pixel space, which means the model is forced to abstract away photometric noise and focus on structure.

The architecture consists of three components:

1. **Context encoder** — maps a temporal sequence of frames into a latent state sequence.
2. **Predictor** — takes the current state sequence and predicts the next state in latent space.
3. **Regulariser** — prevents the trivial solution where all representations collapse to a constant.

At inference time, the encoder's spatial features are fed through a lightweight segmentation head that produces a per-pixel binary mask.

---

## Component Breakdown

### ResNet5 — The Encoder

```
ResNet5(in_d=dobs, h_d=henc, out_d=dstc, s1=1, s2=1, s3=1)
```

A 5-layer residual network with **all strides set to 1**. This is a deliberate design choice: no spatial downsampling occurs, so the output feature map has the same spatial resolution as the input. This is critical because the segmentation head needs spatially-aligned features.

Architecture:

```
Conv2d(in_d → henc, 3×3, stride=1, BN, ReLU)
  └─ ResidualBlock(henc → henc,   stride=s1)
  └─ ResidualBlock(henc → 2*henc, stride=s2)
  └─ ResidualBlock(2*henc → dstc, stride=s3)
```

Each `ResidualBlock` is a standard pre-activation residual unit:

```
Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → (+shortcut) → ReLU
```

The shortcut uses a 1×1 Conv2d to match channels when they differ.

**TemporalBatchMixin**: ResNet5 inherits `TemporalBatchMixin`, which allows it to process either 4-D `(B, C, H, W)` or 5-D `(B, C, T, H, W)` tensors transparently. For 5-D input it reshapes to `(B*T, C, H, W)`, runs the convolution, then reshapes back to `(B, C, T, H, W)`. This enables frame-by-frame encoding without writing separate loops.

The encoder is used as **both** the context encoder and the "target encoder" in the JEPA framework (they share weights, rather than using an EMA target encoder as in some JEPA variants).

---

### ResUNet — The Predictor

```
ResUNet(in_d=2*dstc, h_d=hpre, out_d=dstc)
```

A U-Net with residual blocks, used inside `StateOnlyPredictor`. Takes a pair of consecutive latent states (concatenated along the channel axis) and predicts the next state.

Architecture:

```
Encoder path:
  Conv2d(2*dstc → hpre)
  enc1: ResBlock(hpre → hpre,    stride=1)
  enc2: ResBlock(hpre → 2*hpre,  stride=2)   ← ×2 spatial downsampling
  enc3: ResBlock(2*hpre → 4*hpre,stride=2)   ← ×4 total
  bott: ResBlock(4*hpre → 8*hpre,stride=2)   ← ×8 total

Decoder path (with skip connections):
  up3: ConvTranspose2d(8*hpre → 4*hpre) + cat(enc3) → dec3
  up2: ConvTranspose2d(4*hpre → 2*hpre) + cat(enc2) → dec2
  up1: ConvTranspose2d(2*hpre → hpre)   + cat(enc1) → dec1
  head: Conv2d(hpre → dstc, 1×1)
```

The `_match_size` method handles any spatial mismatches from odd-sized inputs via bilinear interpolation, making the architecture resolution-agnostic.

---

### StateOnlyPredictor

```python
class StateOnlyPredictor(nn.Module):
    def forward(self, x, a=None):
        prev_state = x[:, :, :-1]    # all frames except the last
        next_state = x[:, :, 1:]     # all frames except the first
        combined = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined)
```

A thin wrapper around `ResUNet` that implements the "predict next state from current and previous" protocol without any action conditioning. It:

1. Slices the input state sequence into consecutive pairs.
2. Concatenates them along the channel axis (doubling `dstc` → `2*dstc`).
3. Passes through the ResUNet.
4. Returns a state sequence of the same length.

This is called in the JEPA unroll loop, where the output at each step is used both as the prediction target cost and as the context for the next step (in parallel mode).

---

### JEPA — The Full Model

```python
class JEPA(nn.Module):
    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        ...
    def unroll(self, observations, actions=None, nsteps=1, unroll_mode="parallel", ...):
        ...
```

`unroll()` is the core training method. Two modes are supported:

**Parallel mode** (used in training):

```
state = encoder(observations)           # (B, dstc, T, H, W)
for step in range(nsteps):
    predicted = predictor(state)        # predict next states from current
    predicted = cat(state[:first], predicted[:T-1])   # prepend ground-truth context
    ploss += predcost(state, predicted) / nsteps      # MSE in latent space
regularise(state)                       # VCLoss on full state sequence
```

Each unroll step shifts the context by one frame. The prediction target is always the ground-truth encoder output (not a target encoder), making training simpler at the cost of potential shortcut learning. The `nsteps` parameter controls how many steps of rollout are trained simultaneously.

**Autoregressive mode** (available but not used in current training):

Feeds predicted states back as context for the next step — proper temporal rollout without access to ground-truth states.

---

### Segmentation Head

```python
def build_seg_head(dstc: int, hidden: int = 16) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(dstc, hidden, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden, 1, 1),
        nn.Sigmoid(),
    )
```

A two-layer 1×1 convolution network (with a 3×3 first layer for local context) that maps the `dstc`-dimensional feature map to a single-channel probability map in [0, 1]. Thresholding at 0.5 gives the binary segmentation mask.

During fine-tuning, the encoder output is **temporally averaged** (`enc_state.mean(dim=2)`) before being fed to the seg head. This collapses the T-dimension and gives a single summary feature map per video clip, which is then compared to the 2-D ground-truth mask.

---

### Loss Functions

#### VCLoss (Variance-Covariance Regulariser)

```
VCLoss(std_coeff=10.0, cov_coeff=100.0)
```

Prevents representation collapse without a target encoder. Applied to the encoder output flattened to `(B*T*H*W, dstc)`.

Two terms:

**HingeStdLoss** — penalises any dimension whose standard deviation falls below 1.0:
```
L_std = mean(ReLU(1 - std(z, dim=batch)))
```
This ensures no single dimension collapses to a constant.

**CovarianceLoss** — penalises correlation between embedding dimensions:
```
C = (z^T z) / (N - 1)
L_cov = mean(off_diagonal(C)²)
```
This pushes the covariance matrix toward identity, decorrelating the learned features.

The total regulariser is:
```
L_VC = std_coeff * L_std + cov_coeff * L_cov
```

#### SquareLossSeq (Prediction Loss)

```
L_pred = MSE(encoder(observations), predictor_output)
```

Applied after the projector (same projector as used in VCLoss) to put both targets in the same projected space. Measuring MSE in projected latent space rather than raw encoder space is a form of "stop-gradient on the target" that has been shown empirically to help with stability.

#### Total JEPA Loss

```
L_total = L_VC + L_pred
```

The regulariser dominates early in training (preventing collapse) while the prediction loss guides the encoder to learn temporal structure.

#### BCELoss (Fine-tuning)

During fine-tuning, only binary cross-entropy is used between the seg head output and the binary mask ground truth. The JEPA loss is not computed during fine-tuning.

---

## Projector

```
Projector("dstc-4*dstc-4*dstc")
```

A 3-layer MLP with BatchNorm and ReLU activations, mapping `dstc → 4*dstc → 4*dstc`. The final layer has no bias. Shared between VCLoss and SquareLossSeq — both compute their respective losses in projected space.

---

## Factory Functions

All model construction goes through two factory functions to guarantee consistent architecture configs:

```python
def build_jepa(arch: dict, device: torch.device) -> JEPA:
    encoder   = ResNet5(dobs, henc, dstc)
    predictor = StateOnlyPredictor(ResUNet(2*dstc, hpre, dstc))
    projector = Projector(f"{dstc}-{dstc*4}-{dstc*4}")
    regularizer = VCLoss(10.0, 100.0, proj=projector)
    ploss_fn  = SquareLossSeq(projector)
    return JEPA(encoder, encoder, predictor, regularizer, ploss_fn)

def build_seg_head(dstc: int, hidden: int = 16) -> nn.Sequential:
    ...
```

The `arch` dict carries exactly the fields in `H1Config.arch_dict()`:

| Key | Meaning | Default |
| --- | ------- | ------- |
| `dobs` | Input channels (1 for grayscale TIFF) | 1 |
| `henc` | Encoder hidden channels | 32 |
| `hpre` | Predictor hidden channels | 32 |
| `dstc` | Latent state channels | 8 |
| `seg_head_hidden` | Segmentation head hidden channels | 16 |

This dict is embedded in every compound checkpoint so the architecture can be reconstructed from the file alone.

---

## Checkpoint Format

Two types of checkpoint are written:

**Pretrain checkpoint** — encoder weights only, plain `state_dict`. Used by H1 fine-tuning to warm-start the full JEPA. Not shown in the inference CLI.

```json
{ "model_name": "jepa_pretrained_h1", "run_id": "...", "date": "...", "hypothesis": "H1", "mode": "pretrain" }
```

**Compound checkpoint** — full JEPA + seg head, structured payload. Required for inference. Identified by `"compound": true` in the JSON sidecar.

```python
{
  "type": "neuroseg_jepa_v1",
  "arch": { "dobs": 1, "henc": 32, "hpre": 32, "dstc": 8, "seg_head_hidden": 16 },
  "jepa": <state_dict>,
  "seg_head": <state_dict>,
}
```

Sidecar JSON:
```json
{
  "model_name": "jepa_h1_finetune_f100",
  "run_id": "...",
  "date": "...",
  "compound": true,
  "dice": 0.8821,
  "miou": 0.7943
}
```

At inference time `list_checkpoints()` scans the output directory, filters to `compound=true` entries, and presents them in the interactive CLI with model name, date, Dice, and mIoU for the user to choose from.
