# Experimental Hypotheses

Three experiments are implemented, each testing a distinct property of self-supervised JEPA representations for calcium imaging data. All three share the same encoder architecture (`ResNet5`) and JEPA training objective, differing only in how data is split, what comparisons are drawn, and what metrics are reported.

---

## H1 — Semi-Supervised Segmentation

### Scientific Hypothesis

Self-supervised JEPA pretraining on **unlabeled** calcium imaging data yields a better initialisation for neuron segmentation than random weights, and this advantage grows larger as the amount of labeled training data shrinks.

### Why This Matters

Manual annotation of calcium imaging recordings is the bottleneck for building segmentation systems. If pretraining on the large existing corpus of unlabeled recordings reduces the annotation burden by 10×, that has practical impact on neuroscience pipelines.

### Protocol

```
Unlabeled data (all frames, no annotations)
        │
   JEPA pretraining  ──► pretrained_checkpoint.pt
        │
        ▼
  Labeled data at fractions f ∈ {1%, 5%, 10%, 100%}
        │
  ┌─────┴─────────────────────┐
  │                           │
  Fine-tune from pretrained   Train from scratch
  (mode="finetune")           (mode="supervised_baseline")
  │                           │
  └──────────┬────────────────┘
             ▼
     Compare Dice / mIoU at each fraction f
```

Both branches use identical architecture, hyperparameters, and labeled splits (same random seed). The only difference is whether the encoder weights start from the pretrained JEPA or from random initialisation.

### Fine-Tuning Details

After loading pretrained JEPA weights, the fine-tuning optimizer uses two learning rate groups:
- **Encoder** (`jepa.encoder`): `lr / 10` — slow updates to preserve learned representations.
- **Seg head** (`seg_head`): `lr` — fast updates since it is randomly initialised.

The encoder is **not frozen** — it can adapt to the labeled data. This is true fine-tuning, not a linear probe.

### Expected Result

The pretrained model should outperform the supervised baseline especially at low labeled fractions (1%, 5%). At 100% labeled data, the gap narrows because the baseline has enough data to learn from scratch. The "performance vs labeled fraction" curve should show that pretraining is most valuable when labels are scarce.

### MLflow Tags

| Tag | Values |
| --- | ------ |
| `hypothesis` | `H1` |
| `mode` | `pretrain`, `finetune`, `supervised_baseline` |
| `labeled_fraction` | `0.01`, `0.05`, `0.10`, `1.0` |

### Output Checkpoints

| File pattern | Content |
| ------------ | ------- |
| `jepa_pretrained_h1_<run_id>.pt` | Encoder state dict (pretraining only) |
| `jepa_h1_finetune_f<P>_<run_id>.pt` | Compound: JEPA + seg head (fine-tune at fraction P%) |
| `jepa_h1_supervised_f<P>_<run_id>.pt` | Compound: JEPA + seg head (baseline at fraction P%) |

---

## H2 — Cross-Organism Transfer

### Scientific Hypothesis

JEPA representations learned from one organism's calcium imaging data transfer better to a different organism than supervised representations trained only on the source organism, as measured by the performance drop in segmentation Dice score.

### Why This Matters

Some model organisms (zebrafish, Drosophila) have abundant optical access for calcium imaging but sparse manual annotations. Mouse cortex has the most annotations. If self-supervised representations from mouse cortex transfer to zebrafish with less degradation than a fully-supervised mouse model, practitioners can leverage existing mouse annotations to bootstrap segmentation for less-annotated species.

### Protocol

```
Source organism data (unlabeled)
        │
   JEPA pretraining on source  ──► pretrained_checkpoint.pt
        │
        ▼
Target organism data (labeled, full)
        │
  ┌─────┴─────────────────────────────┐
  │                                   │
  Fine-tune from pretrained            Train from scratch
  (mode="finetune", f=1.0)            (mode="supervised_baseline", f=1.0)
  │                                   │
  └──────────┬────────────────────────┘
             ▼
  Compare Dice / mIoU on target organism held-out test data
```

The key metric is not absolute performance but the **transfer gap**: how much does performance drop compared to a model trained directly on the target organism? Smaller gap = better generalisation.

### Organism-Agnostic Design

H2 does not hardcode "zebrafish → mouse" or any specific organism pair. Instead:

- The CLI takes `--source-data` and `--target-data` flags pointing to any two Neurofinder directories.
- The organism label is inferred from the Neurofinder directory name: `neurofinder.04.*` → `zebrafish`, `neurofinder.00.*`–`03.*` → `mouse.visual_cortex`, `neurofinder.10.*` → `mouse.hippocampus`.
- If a directory does not match the known Neurofinder naming convention, the directory basename is used as the organism label.

```python
_NF_ORGANISM_MAP = {
    "00": "mouse.visual_cortex",
    "01": "mouse.visual_cortex",
    "02": "mouse.visual_cortex",
    "03": "mouse.visual_cortex",
    "04": "zebrafish",
    "10": "mouse.hippocampus",
}
```

This makes the organism metadata in MLflow descriptive and reproducible without manual tagging.

### Natural Data Split (Neurofinder)

A sensible H2 configuration using the bundled Neurofinder datasets:

| Role | Datasets | Organism |
| ---- | -------- | -------- |
| Source | `neurofinder.04.*` | Zebrafish |
| Target | `neurofinder.00.*`–`03.*` | Mouse visual cortex |

Or in reverse (mouse → zebrafish) for the opposite transfer direction.

### MLflow Tags

| Tag | Values |
| --- | ------ |
| `hypothesis` | `H2` |
| `source_organism` | Inferred (e.g. `zebrafish`) |
| `target_organism` | Inferred (e.g. `mouse.visual_cortex`) |
| `mode` | `pretrain`, `finetune`, `supervised_baseline` |

### CLI

```bash
uv run main.py \
  --mode train --H2 \
  --data /path/to/source \
  --output ./checkpoints \
  --source-data /path/to/neurofinder.04 \
  --target-data /path/to/neurofinder.00 \
  --pretrain-epochs 100 \
  --finetune-epochs 10
```

The `--finetune-epochs` is intentionally small (default 10) to model a "limited fine-tuning budget" scenario — the target organism has limited compute or annotation time available.

---

## H3 — Temporal Representation Stability

### Scientific Hypothesis

JEPA pretraining produces encoder representations where the same neuron has more consistent embedding across time than different neurons at the same time point. This property — temporal identity stability — is larger for JEPA-pretrained encoders than for supervised or randomly-initialised baselines.

### Why This Matters

Stable per-neuron representations are a necessary property for any downstream analysis that tracks neurons over time (cell tracking, drift correction, session-to-session alignment). If JEPA representations exhibit this property as an emergent consequence of predicting future states, it validates the approach as a foundation for temporal analysis beyond segmentation.

### Protocol

For each of three encoder configurations:

| Configuration | Checkpoint source |
| ------------- | ----------------- |
| `pretrained` | H1 pretraining output |
| `supervised_baseline` | H1 supervised baseline (fine-tuned, 100%) |
| `no_pretrain` | Random initialisation, same architecture |

The following metrics are computed per configuration:

1. **Encode** all frames in the labeled dataset.
2. **Pool per-neuron embeddings**: For each neuron `n` and frame `t`, spatially average the encoder feature map over the pixels belonging to that neuron (from the integer-labelled mask).
3. **Within-neuron similarity**: For each pair of frames `(t1, t2)` and the same neuron `n`, compute cosine similarity between its embedding at `t1` and `t2`.
4. **Between-neuron similarity**: For each pair of distinct neurons `(n1, n2)` at the same frame `t`, compute cosine similarity between their embeddings.
5. **Gap**: `within_sim − between_sim`.

A large gap means the model's representations are more discriminative in time than in space — the encoding of a neuron over time is more consistent than the differences between neurons. This is the desired property for neuron tracking.

### Implementation Note

H3 requires **integer-labelled masks** (not binary), since it needs to identify which pixels belong to which neuron. Both `NeurofinderDataset` and `LabeledTIFFDataset` support this via `binarize=False`.

H3 reads the `pretrained_ckpt` and `supervised_ckpt` paths from config. The `no_pretrain` condition always runs automatically — no checkpoint needed.

### Prerequisite

H3 is a **post-hoc analysis**, not a standalone training run. It consumes checkpoints produced by H1. The recommended workflow:

```bash
# 1. Run H1 to produce pretrained and supervised-baseline checkpoints
uv run main.py --mode train --H1 \
  --data /path/to/neurofinder \
  --output ./checkpoints \
  --labeled-data /path/to/neurofinder

# 2. Run H3 pointing at those checkpoints via config
uv run main.py --mode train --H3 \
  --data /path/to/neurofinder \
  --output ./checkpoints \
  --config config.yaml
```

Where `config.yaml` contains:
```yaml
h3_data_dir: /path/to/neurofinder
pretrained_ckpt: ./checkpoints/jepa_pretrained_h1_<run_id>.pt
supervised_ckpt: ./checkpoints/jepa_h1_supervised_f100_<run_id>.pt
```

### Expected Result

JEPA-pretrained encoders should exhibit a larger within-vs-between gap than the supervised baseline, which in turn should be larger than the random (no_pretrain) baseline. If the gap is small or inverted for the pretrained model, this would indicate that prediction-in-latent-space does not induce temporal identity structure for these data.

### MLflow Tags

| Tag | Values |
| --- | ------ |
| `hypothesis` | `H3` |
| `mode` | `pretrained`, `supervised_baseline`, `no_pretrain` |

### MLflow Metrics

| Metric | Description |
| ------ | ----------- |
| `within_sim` | Mean cosine similarity: same neuron, different frames |
| `between_sim` | Mean cosine similarity: different neurons, same frame |
| `gap` | `within_sim − between_sim` (primary metric) |

---

## Comparing Across Hypotheses

| Property tested | H1 | H2 | H3 |
| --------------- | -- | -- | -- |
| Sample efficiency | Yes | No | No |
| Cross-domain transfer | No | Yes | No |
| Temporal identity | No | No | Yes |
| Requires labeled data | Yes (small fraction) | Yes (target organism) | Yes (for mask pooling) |
| Requires multiple runs | Yes (pretrain + finetune × fraction) | Yes (pretrain + finetune + baseline) | Yes (run H1 first) |
| Primary metric | Dice vs fraction curve | Transfer drop (ΔJEPA − ΔBaseline) | Gap = within − between |
