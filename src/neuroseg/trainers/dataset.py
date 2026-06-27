import json
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_tiff(path: str) -> np.ndarray:
    data = tiff.imread(path)
    if data.ndim == 2:
        data = data[np.newaxis]
    elif data.ndim == 4:
        data = data[:, 0] if data.shape[1] < data.shape[0] else data[0]
    return data.astype(np.float32)


def _normalize(data: np.ndarray) -> np.ndarray:
    mn, mx = float(data.min()), float(data.max())
    return (data - mn) / (mx - mn + 1e-8)


def _resize_frames(frames: np.ndarray, img_size: int, interpolation: int) -> np.ndarray:
    return np.stack([
        cv2.resize(frames[t], (img_size, img_size), interpolation=interpolation)
        for t in range(frames.shape[0])
    ])


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def is_neurofinder_dir(path: str | Path) -> bool:
    return (Path(path) / "images").is_dir()


def find_neurofinder_dirs(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if is_neurofinder_dir(root):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and is_neurofinder_dir(d))


# ---------------------------------------------------------------------------
# Neurofinder loaders
# ---------------------------------------------------------------------------

def _load_nf_video(nf_dir: Path) -> np.ndarray:
    img_dir = nf_dir / "images"
    paths = sorted(img_dir.glob("*.tiff")) or sorted(img_dir.glob("*.tif"))
    frames = []
    for p in paths:
        frame = np.array(Image.open(str(p))).astype(np.float32)
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        frames.append(frame)
    return np.stack(frames)


def _build_nf_mask(nf_dir: Path, orig_h: int, orig_w: int, img_size: int) -> np.ndarray:
    """
    Build a pixel-accurate integer-labeled mask from regions.json.
    Each pixel that belongs to neuron N is assigned integer label N (1-indexed).
    Returns an (img_size, img_size) int32 array.
    """
    regions_path = nf_dir / "regions" / "regions.json"
    if not regions_path.exists():
        return np.zeros((img_size, img_size), dtype=np.int32)

    with open(regions_path) as f:
        regions = json.load(f)

    mask = np.zeros((img_size, img_size), dtype=np.int32)
    for neuron_id, region in enumerate(regions, start=1):
        coords = np.array(region["coordinates"])
        xs = np.clip((coords[:, 0] * img_size / orig_w).astype(int), 0, img_size - 1)
        ys = np.clip((coords[:, 1] * img_size / orig_h).astype(int), 0, img_size - 1)
        mask[ys, xs] = neuron_id

    return mask


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class TIFFVideoDataset(Dataset):
    """
    Unlabeled dataset for JEPA self-supervised pretraining.
    Accepts a list of multi-frame TIFF stack paths.
    """

    def __init__(self, file_paths: list[str], seq_len: int, img_size: int):
        self.seq_len = seq_len
        self.img_size = img_size
        self.clips: list[np.ndarray] = []

        for path in file_paths:
            video = _normalize(_load_tiff(path))
            T = video.shape[0]
            if T < seq_len:
                padded = np.zeros((seq_len, *video.shape[1:]), dtype=video.dtype)
                padded[:T] = video
                self.clips.append(padded)
            else:
                for start in range(0, T - seq_len + 1, seq_len):
                    self.clips.append(video[start : start + seq_len])

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        if clip.shape[-1] != self.img_size or clip.shape[-2] != self.img_size:
            clip = _resize_frames(clip, self.img_size, cv2.INTER_LINEAR)
        return {"video": torch.from_numpy(clip).unsqueeze(0).float()}


class LabeledTIFFDataset(Dataset):
    """
    Labeled dataset for fine-tuning and evaluation.

    Expected layout::

        data_dir/
            sample_001/
                video.tif   — multi-frame TIFF stack  (T × H × W)
                mask.tif    — integer label mask       (H × W) or (T × H × W)
            sample_002/
                ...

    Parameters
    ----------
    binarize : bool
        True → threshold mask to {0, 1} for BCELoss fine-tuning.
        False → preserve integer neuron IDs for H3 similarity analysis.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        img_size: int,
        labeled_fraction: float = 1.0,
        seed: int = 0,
        binarize: bool = True,
    ):
        self.seq_len = seq_len
        self.img_size = img_size
        self.binarize = binarize

        samples = sorted(Path(data_dir).iterdir())
        rng = np.random.default_rng(seed)
        n = max(1, int(len(samples) * labeled_fraction))
        chosen = rng.choice(len(samples), size=n, replace=False)
        self.samples = [samples[i] for i in sorted(chosen)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]
        video = _normalize(_load_tiff(str(sample_dir / "video.tif")))
        mask = _load_tiff(str(sample_dir / "mask.tif"))

        if mask.ndim == 2:
            mask = np.stack([mask] * video.shape[0])

        T = min(video.shape[0], self.seq_len)
        video = video[:T]
        mask = mask[:T]

        if video.shape[-1] != self.img_size or video.shape[-2] != self.img_size:
            video = _resize_frames(video, self.img_size, cv2.INTER_LINEAR)
            mask = _resize_frames(mask, self.img_size, cv2.INTER_NEAREST)

        video_t = torch.from_numpy(video).unsqueeze(0).float()
        mask_t = (
            torch.from_numpy((mask > 0).astype(np.float32))
            if self.binarize
            else torch.from_numpy(mask.astype(np.int64))
        )
        return {"video": video_t, "mask": mask_t}


class NeurofinderDataset(Dataset):
    """
    Reads Neurofinder-format data::

        nf_dir/
            images/
                image00000.tiff
                image00001.tiff
                ...
            regions/
                regions.json    ← pixel-coordinate annotations (one list per neuron)

    ``data_dir`` can be either a single Neurofinder directory (contains ``images/``)
    or a parent directory whose subdirectories are Neurofinder datasets — all are
    loaded and pooled into a single set of temporal clips.

    Parameters
    ----------
    labeled : bool
        False → skip regions.json; returns only ``{"video": ...}`` (pretraining mode).
        True  → load pixel-accurate mask; returns ``{"video": ..., "mask": ...}``.
    binarize : bool
        Only used when ``labeled=True``.
        True  → binary {0,1} mask (BCELoss fine-tuning).
        False → integer neuron-ID mask (H3 cosine-similarity analysis).
    labeled_fraction : float
        Fraction of temporal clips to keep, sampled reproducibly.
        Useful for H1's labeled-fraction ablation.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        img_size: int,
        labeled: bool = True,
        labeled_fraction: float = 1.0,
        seed: int = 0,
        binarize: bool = True,
    ):
        self.seq_len = seq_len
        self.img_size = img_size
        self.labeled = labeled
        self.binarize = binarize

        nf_dirs = find_neurofinder_dirs(data_dir)
        if not nf_dirs:
            raise ValueError(f"No Neurofinder directories found under {data_dir}")

        self.clips: list[tuple[np.ndarray, np.ndarray | None]] = []

        for nf_dir in nf_dirs:
            raw = _load_nf_video(nf_dir)
            orig_h, orig_w = raw.shape[1], raw.shape[2]

            video = _normalize(raw)
            if video.shape[-1] != img_size or video.shape[-2] != img_size:
                video = _resize_frames(video, img_size, cv2.INTER_LINEAR)

            mask = _build_nf_mask(nf_dir, orig_h, orig_w, img_size) if labeled else None

            T = video.shape[0]
            if T < seq_len:
                padded = np.zeros((seq_len, *video.shape[1:]), dtype=video.dtype)
                padded[:T] = video
                self.clips.append((padded, mask))
            else:
                for start in range(0, T - seq_len + 1, seq_len):
                    self.clips.append((video[start : start + seq_len], mask))

        if labeled_fraction < 1.0:
            rng = np.random.default_rng(seed)
            n = max(1, int(len(self.clips) * labeled_fraction))
            idx = rng.choice(len(self.clips), size=n, replace=False)
            self.clips = [self.clips[i] for i in sorted(idx)]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip, mask = self.clips[idx]
        video_t = torch.from_numpy(clip).unsqueeze(0).float()

        if mask is None or not self.labeled:
            return {"video": video_t}

        mask_seq = np.stack([mask] * self.seq_len)
        mask_t = (
            torch.from_numpy((mask_seq > 0).astype(np.float32))
            if self.binarize
            else torch.from_numpy(mask_seq.astype(np.int64))
        )
        return {"video": video_t, "mask": mask_t}
