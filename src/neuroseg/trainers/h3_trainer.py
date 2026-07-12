from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from neuroseg.checkpoint import load_compound_checkpoint
from neuroseg.models.state import State
from neuroseg.trainers.dataset import (
    LabeledTIFFDataset,
    NeurofinderDataset,
    find_neurofinder_dirs,
    is_neurofinder_dir,
)
from neuroseg.trainers.h1_trainer import H1Config, build_config, setup_seed
from neuroseg.trainers.jepa import JEPA, build_jepa


def _load_encoder(checkpoint_path: Optional[str], cfg: H1Config, device: torch.device) -> JEPA:
    """Load a JEPA encoder from a checkpoint, or build a random one if path is None."""
    if checkpoint_path is None:
        return build_jepa(cfg.arch_dict(), device)

    import json as _json
    path = Path(checkpoint_path)
    payload = torch.load(str(path), map_location=device, weights_only=False)

    if isinstance(payload, dict) and payload.get("type") == "neuroseg_jepa_v1":
        arch = payload.get("arch", cfg.arch_dict())
        jepa = build_jepa(arch, device)
        jepa.load_state_dict(payload["jepa"], strict=False)
    else:
        arch = cfg.arch_dict()
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            sidecar_meta = _json.loads(sidecar.read_text())
            if "arch" in sidecar_meta:
                arch = sidecar_meta["arch"]
        jepa = build_jepa(arch, device)
        jepa.load_state_dict(payload, strict=False)

    return jepa


def _normalize_rows(a: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D array so dot products give cosine similarity."""
    a = np.asarray(a, dtype=np.float32)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def _upper_triangle_sims(vecs: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine similarities (strict upper triangle) of the row vectors."""
    v = _normalize_rows(vecs)
    sims = v @ v.T
    iu = np.triu_indices(len(v), k=1)
    return sims[iu]


@torch.inference_mode()
def _compute_similarity_gap(
    jepa: JEPA,
    dataset: Dataset,
    device: torch.device,
    img_size: int,
    max_clips: Optional[int] = None,
) -> dict:
    """
    Pool a per-neuron embedding (encoder features averaged over the neuron's footprint)
    for every neuron in every frame, then measure:
      - within_sim  : cosine similarity of the same neuron across different frames
      - between_sim : cosine similarity of different neurons within the same frame
    Similarities use vectorized matrix products. When max_clips is set, clips are
    sampled evenly across the recording to bound runtime. Returns the means and the
    gap (within - between); larger gap = more stable, discriminative features.
    """
    jepa.eval()
    n = len(dataset)
    if max_clips and max_clips < n:
        indices = np.linspace(0, n - 1, max_clips).astype(int)
    else:
        indices = np.arange(n)

    per_neuron: dict = {}
    per_frame: list = []

    for ci in indices:
        sample = dataset[int(ci)]
        x = sample["video"].unsqueeze(0).to(device)
        mask = sample["mask"].numpy().astype(np.int64)
        enc = jepa.encoder(x).squeeze(0).cpu().numpy()

        for t in range(enc.shape[1]):
            frame_mask = mask[t]
            frame_embs = []
            for neuron_id in np.unique(frame_mask):
                if neuron_id <= 0:
                    continue
                region = frame_mask == neuron_id
                if region.sum() == 0:
                    continue
                vec = enc[:, t, region].mean(axis=-1)
                per_neuron.setdefault(int(neuron_id), []).append(vec)
                frame_embs.append(vec)
            if len(frame_embs) >= 2:
                per_frame.append(np.stack(frame_embs))

    within = [_upper_triangle_sims(np.stack(v)) for v in per_neuron.values() if len(v) >= 2]
    between = [_upper_triangle_sims(f) for f in per_frame]

    if not within or not between:
        return {"within_sim": float("nan"), "between_sim": float("nan"), "gap": float("nan")}

    within_mean = float(np.concatenate(within).mean())
    between_mean = float(np.concatenate(between).mean())
    return {"within_sim": within_mean, "between_sim": between_mean, "gap": within_mean - between_mean}


def _run_mode(
    mode: str,
    checkpoint_path: Optional[str],
    dataset: LabeledTIFFDataset,
    cfg: H1Config,
    device: torch.device,
    log_path: Path,
    max_clips: Optional[int] = None,
):
    """Load an encoder, compute the similarity gap for the given mode, and log the results."""
    from neuroseg.logger import RunLogger
    jepa = _load_encoder(checkpoint_path, cfg, device)
    metrics = _compute_similarity_gap(jepa, dataset, device, cfg.img_size, max_clips=max_clips)

    print(
        f"[H3/{mode}] within={metrics['within_sim']:.4f}  "
        f"between={metrics['between_sim']:.4f}  gap={metrics['gap']:.4f}"
    )

    logger = RunLogger(log_path, hypothesis="H3", mode=mode, model_name=mode)
    logger.log(**metrics)


def run_h3(state: State):
    """
    H3 — Temporal representation stability.

    Protocol
    --------
    For each encoder mode (pretrained / supervised_baseline / no_pretrain):
      1. Encode all frames in the labeled dataset.
      2. Pool per-neuron embeddings using the integer segmentation masks.
      3. Compute within-neuron cosine similarity (same neuron, different frames).
      4. Compute between-neuron cosine similarity (different neurons, same frame).
      5. Report the gap: within − between.  Larger gap = more stable representations.

    Required config keys
    --------------------
    h3_data_dir      : str  — labeled data directory (LabeledTIFFDataset layout).
    pretrained_ckpt  : str  — path to pretrained JEPA checkpoint  (optional).
    supervised_ckpt  : str  — path to supervised-baseline checkpoint (optional).

    At least one of the three modes will always run (no_pretrain needs no checkpoint).

    Logging
    -------
    Results are appended to <output>/logs/runs.csv with hypothesis=H3 and
    mode in {pretrained, supervised_baseline, no_pretrain}.
    """
    cfg = build_config(state)
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extra = state.get("config", {})
    h3_data_dir = extra.get("h3_data_dir", state.get("data_dir", ""))

    if not h3_data_dir or not Path(h3_data_dir).exists():
        raise ValueError(
            "H3 requires labeled data at 'h3_data_dir' in config. "
            "Pass it via --data or add 'h3_data_dir' to your config."
        )

    if is_neurofinder_dir(h3_data_dir) or find_neurofinder_dirs(h3_data_dir):
        dataset = NeurofinderDataset(
            h3_data_dir,
            seq_len=cfg.seq_len,
            img_size=cfg.img_size,
            labeled=True,
            labeled_fraction=1.0,
            seed=cfg.seed,
            binarize=False,
        )
    else:
        dataset = LabeledTIFFDataset(
            h3_data_dir,
            seq_len=cfg.seq_len,
            img_size=cfg.img_size,
            labeled_fraction=1.0,
            seed=cfg.seed,
            binarize=False,
        )

    pretrained_ckpt = extra.get("pretrained_ckpt")
    supervised_ckpt = extra.get("supervised_ckpt")
    max_clips = extra.get("h3_max_clips")

    log_path = Path(state["output_dir"]) / "logs" / "runs.csv"
    n_used = min(len(dataset), max_clips) if max_clips else len(dataset)
    print(f"[H3] device={device} | clips={len(dataset)} | using {n_used} (h3_max_clips={max_clips})")

    if pretrained_ckpt:
        _run_mode("pretrained", pretrained_ckpt, dataset, cfg, device, log_path, max_clips)

    for enc in (extra.get("h3_extra_encoders") or []):
        _run_mode(enc["mode"], enc["ckpt"], dataset, cfg, device, log_path, max_clips)

    if supervised_ckpt:
        _run_mode("supervised_baseline", supervised_ckpt, dataset, cfg, device, log_path, max_clips)

    _run_mode("no_pretrain", None, dataset, cfg, device, log_path, max_clips)

    print("[H3] Done.")

    from neuroseg.plots import plot_h3_similarity
    plot_h3_similarity(log_path, Path(state["output_dir"]) / "figures")
