"""
H3 — Learned representations are more stable over time for the same neuron.

Protocol:
1. Load a pretrained (or randomly initialised) JEPA encoder.
2. For each video, extract per-neuron embeddings across frames using segmentation masks
   (cosine-pooling: weighted average of encoder features over the neuron's pixel region).
3. Compute within-neuron cosine similarity (same neuron, different frames) and
   between-neuron cosine similarity (different neurons, same frame).
4. Report the within–between gap; larger gap = more temporally stable representations.
5. Repeat for pretrained, supervised-baseline, and no-pretrain encoders.

Required config keys:
    h3_data_dir        : str  — path to annotated TIFF data (LabeledTIFFDataset layout)
    pretrained_ckpt    : str  — path to pretrained JEPA checkpoint (optional)
    supervised_ckpt    : str  — path to supervised-baseline checkpoint (optional)

MLflow tags: hypothesis=H3, mode={pretrained|supervised_baseline|no_pretrain}
"""

from neuroseg.models.state import State


def run_h3(state: State):
    raise NotImplementedError(
        "H3 trainer is not yet implemented. "
        "Provide h3_data_dir (and optionally pretrained_ckpt / supervised_ckpt) in config."
    )
