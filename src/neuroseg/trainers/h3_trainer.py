from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


@torch.inference_mode()
def _compute_similarity_gap(
    jepa: JEPA,
    dataset: Dataset,
    device: torch.device,
    img_size: int,
) -> dict:
    """
    For each sample compute per-neuron embeddings across frames, then measure:
      - within_sim  : cosine similarity between the same neuron at different frames
      - between_sim : cosine similarity between different neurons at the same frame
    Returns mean values and the gap (within - between).
    """
    jepa.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_within, all_between = [], []

    for batch in loader:
        x = batch["video"].to(device)
        mask = batch["mask"].squeeze(0).numpy().astype(np.int64)

        enc = jepa.encoder(x)
        enc = enc.squeeze(0).cpu().numpy()

        T = enc.shape[1]
        neuron_ids = [n for n in np.unique(mask) if n > 0]
        if len(neuron_ids) < 2:
            continue

        embeddings = {}
        for n in neuron_ids:
            embs = []
            for t in range(T):
                region = mask[t] == n
                if region.sum() == 0:
                    continue
                feat = enc[:, t, region].mean(axis=-1)
                embs.append(feat)
            if embs:
                embeddings[n] = np.stack(embs)

        valid_neurons = [n for n in neuron_ids if n in embeddings]
        if len(valid_neurons) < 2:
            continue

        for n in valid_neurons:
            embs = embeddings[n]
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    a = torch.from_numpy(embs[i]).float()
                    b = torch.from_numpy(embs[j]).float()
                    all_within.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

        for t in range(T):
            frame_embs = {}
            for n in valid_neurons:
                region = mask[t] == n
                if region.sum() == 0:
                    continue
                frame_embs[n] = enc[:, t, region].mean(axis=-1)

            neurons_in_frame = list(frame_embs.keys())
            for i in range(len(neurons_in_frame)):
                for j in range(i + 1, len(neurons_in_frame)):
                    a = torch.from_numpy(frame_embs[neurons_in_frame[i]]).float()
                    b = torch.from_numpy(frame_embs[neurons_in_frame[j]]).float()
                    all_between.append(
                        F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                    )

    if not all_within or not all_between:
        return {"within_sim": float("nan"), "between_sim": float("nan"), "gap": float("nan")}

    within = float(np.mean(all_within))
    between = float(np.mean(all_between))
    return {"within_sim": within, "between_sim": between, "gap": within - between}


def _run_mode(
    mode: str,
    checkpoint_path: Optional[str],
    dataset: LabeledTIFFDataset,
    cfg: H1Config,
    device: torch.device,
    log_path: Path,
):
    """Load an encoder, compute the similarity gap for the given mode, and log the results."""
    from neuroseg.logger import RunLogger
    jepa = _load_encoder(checkpoint_path, cfg, device)
    metrics = _compute_similarity_gap(jepa, dataset, device, cfg.img_size)

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

    log_path = Path(state["output_dir"]) / "logs" / "runs.csv"
    print(f"[H3] device={device} | samples={len(dataset)}")

    if pretrained_ckpt:
        _run_mode("pretrained", pretrained_ckpt, dataset, cfg, device, log_path)

    if supervised_ckpt:
        _run_mode("supervised_baseline", supervised_ckpt, dataset, cfg, device, log_path)

    _run_mode("no_pretrain", None, dataset, cfg, device, log_path)

    print("[H3] Done.")

    from neuroseg.plots import plot_h3_similarity
    plot_h3_similarity(log_path, Path(state["output_dir"]) / "figures")
