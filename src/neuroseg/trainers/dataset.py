import json
import os
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
    """Read a TIFF file and return a float32 array with shape (T, H, W)."""
    data = tiff.imread(path)
    if data.ndim == 2:
        data = data[np.newaxis]
    elif data.ndim == 4:
        data = data[:, 0] if data.shape[1] < data.shape[0] else data[0]
    return data.astype(np.float32)


def _normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    mn, mx = float(data.min()), float(data.max())
    return (data - mn) / (mx - mn + 1e-8)


def _augment_clip(clip: np.ndarray) -> np.ndarray:
    """
    Apply a random, temporally-consistent augmentation to a (T, H, W) clip in [0, 1].

    Spatial ops (random resized crop, horizontal/vertical flip, 90-degree rotation)
    use the SAME parameters for every frame so temporal correspondence is preserved;
    photometric jitter and additive Gaussian noise decorrelate the view from the raw
    clip. Used only for self-supervised pretraining, to prevent the encoder from
    memorizing clips (overfitting) and to force invariant, non-collapsed features.
    """
    rng = np.random.default_rng()
    out = clip.astype(np.float32)
    _, h, w = out.shape

    if rng.random() < 0.8:
        scale = rng.uniform(0.6, 1.0)
        ch, cw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        top = int(rng.integers(0, h - ch + 1))
        left = int(rng.integers(0, w - cw + 1))
        cropped = out[:, top:top + ch, left:left + cw]
        out = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in cropped])

    if rng.random() < 0.5:
        out = out[:, ::-1, :]
    if rng.random() < 0.5:
        out = out[:, :, ::-1]
    if h == w:
        k = int(rng.integers(0, 4))
        if k:
            out = np.rot90(out, k, axes=(1, 2))

    out = out * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1)
    out = out + rng.normal(0.0, 0.02, size=out.shape)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32))


class AugmentedClips(Dataset):
    """Training-only wrapper that applies _augment_clip to a base dataset's 'video' clips."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        """Return the number of clips in the wrapped dataset."""
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        """Return the base sample with its video clip randomly augmented."""
        sample = self.base[idx]
        clip = sample["video"].squeeze(0).numpy()
        aug = _augment_clip(clip)
        return {"video": torch.from_numpy(aug).unsqueeze(0).float()}


def _iter_frames(path: str):
    """Yield frames as 2-D float32 arrays from a multi-frame TIFF or Zeiss CZI (.czi/.sec) video."""
    ext = Path(path).suffix.lower()
    if ext in (".czi", ".sec"):
        import czifile
        arr = np.squeeze(czifile.imread(str(path)))
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim > 3:
            arr = arr.reshape((-1,) + arr.shape[-2:])
        for i in range(arr.shape[0]):
            yield np.asarray(arr[i], dtype=np.float32)
    else:
        with tiff.TiffFile(str(path)) as tf:
            n_pages = len(tf.pages)
            if n_pages > 1:
                for page in tf.pages:
                    a = page.asarray()
                    if a.ndim == 3:
                        a = a[..., 0]
                    yield a.astype(np.float32)
                return
        try:
            arr = tiff.memmap(str(path))
        except Exception:
            arr = tiff.imread(str(path))
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim > 3:
            arr = arr.reshape((-1,) + arr.shape[-2:])
        for i in range(arr.shape[0]):
            yield np.asarray(arr[i], dtype=np.float32)


def _resize_frames(frames: np.ndarray, img_size: int, interpolation: int) -> np.ndarray:
    """Resize every frame in a (T, H, W) array to (img_size, img_size)."""
    return np.stack([
        cv2.resize(frames[t], (img_size, img_size), interpolation=interpolation)
        for t in range(frames.shape[0])
    ])


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def is_neurofinder_dir(path: str | Path) -> bool:
    """Return True if path is a Neurofinder directory (contains an images/ subdirectory)."""
    return (Path(path) / "images").is_dir()


def find_neurofinder_dirs(data_dir: str | Path) -> list[Path]:
    """Return all Neurofinder dataset directories at or under data_dir, at any nesting depth."""
    root = Path(data_dir)
    if is_neurofinder_dir(root):
        return [root]
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if "images" in dirnames:
            found.append(Path(dirpath))
            dirnames[:] = []
    return sorted(found)


# ---------------------------------------------------------------------------
# Neurofinder loaders
# ---------------------------------------------------------------------------

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
        ys = np.clip((coords[:, 0] * img_size / orig_h).astype(int), 0, img_size - 1)
        xs = np.clip((coords[:, 1] * img_size / orig_w).astype(int), 0, img_size - 1)
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

    def __init__(self, file_paths: list[str], seq_len: int, img_size: int, clip_stride: int | None = None):
        self.seq_len = seq_len
        self.img_size = img_size
        self.clips: list[np.ndarray] = []
        stride = clip_stride or seq_len

        for path in file_paths:
            video = _normalize(_load_tiff(path))
            T = video.shape[0]
            if T < seq_len:
                padded = np.zeros((seq_len, *video.shape[1:]), dtype=video.dtype)
                padded[:T] = video
                self.clips.append(padded)
            else:
                for start in range(0, T - seq_len + 1, stride):
                    self.clips.append(video[start : start + seq_len])

    def __len__(self) -> int:
        """Return the number of temporal clips in the dataset."""
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        """Return a dict with a 'video' tensor of shape (1, seq_len, H, W)."""
        clip = self.clips[idx]
        if clip.shape[-1] != self.img_size or clip.shape[-2] != self.img_size:
            clip = _resize_frames(clip, self.img_size, cv2.INTER_LINEAR)
        return {"video": torch.from_numpy(clip).unsqueeze(0).float()}


class VideoFolderDataset(Dataset):
    """
    Unlabeled dataset for JEPA pretraining from a folder of raw video files.

    Reads multi-frame TIFF stacks and Zeiss CZI files (``.czi`` / ``.sec``) found at
    any depth under ``data_dir``. Each frame is resized to ``img_size`` on load, so
    memory stays bounded regardless of the source file sizes.

    A stack whose frame width is an exact integer multiple of its height (>1) is
    treated as that many z-planes stacked side-by-side and split into separate
    square videos — e.g. an ETL 6-plane 512×3072 stack becomes six 512×512 videos.
    """

    def __init__(self, data_dir: str, seq_len: int, img_size: int,
                 clip_stride: int | None = None, split_planes: bool = True,
                 max_file_gb: float | None = None):
        self.seq_len = seq_len
        self.img_size = img_size
        stride = clip_stride or seq_len
        exts = (".tif", ".tiff", ".czi", ".sec")
        files = sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts)

        if max_file_gb is not None:
            kept = []
            for p in files:
                gb = p.stat().st_size / 1e9
                if gb > max_file_gb:
                    print(f"[VideoFolderDataset] skipping {p.name} ({gb:.1f} GB > {max_file_gb} GB cap)")
                else:
                    kept.append(p)
            files = kept

        if not files:
            raise ValueError(f"No .tif/.czi/.sec video files found under {data_dir}")

        self.clips: list[np.ndarray] = []
        for f in files:
            planes: list[list[np.ndarray]] | None = None
            plane_w = 0
            for frame in _iter_frames(str(f)):
                h, w = frame.shape[-2], frame.shape[-1]
                if planes is None:
                    n = w // h if (split_planes and w > h and w % h == 0 and w // h > 1) else 1
                    plane_w = w // n
                    planes = [[] for _ in range(n)]
                for pi in range(len(planes)):
                    sub = frame[:, pi * plane_w:(pi + 1) * plane_w]
                    if sub.shape[-1] != img_size or sub.shape[-2] != img_size:
                        sub = cv2.resize(sub, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    planes[pi].append(sub)

            for pv in planes or []:
                if not pv:
                    continue
                video = _normalize(np.stack(pv))
                T = video.shape[0]
                if T < seq_len:
                    padded = np.zeros((seq_len, img_size, img_size), dtype=video.dtype)
                    padded[:T] = video
                    self.clips.append(padded)
                else:
                    for s in range(0, T - seq_len + 1, stride):
                        self.clips.append(video[s:s + seq_len])

    def __len__(self) -> int:
        """Return the number of temporal clips across all video files."""
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        """Return a dict with a 'video' tensor of shape (1, seq_len, H, W)."""
        return {"video": torch.from_numpy(self.clips[idx]).unsqueeze(0).float()}


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
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return a dict with 'video' and 'mask' tensors for the given sample."""
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
    exclude_dirs : list[str | Path] | None
        Neurofinder directories to drop from the pool (resolved-path match).
        Used to hold a recording out of self-supervised pretraining so it stays
        unseen for downstream fine-tuning/testing (H1 leakage control).
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
        clip_stride: int | None = None,
        exclude_dirs: list[str | Path] | None = None,
    ):
        self.seq_len = seq_len
        self.img_size = img_size
        self.labeled = labeled
        self.binarize = binarize
        stride = clip_stride or seq_len

        nf_dirs = find_neurofinder_dirs(data_dir)
        if exclude_dirs:
            excluded = {Path(d).resolve() for d in exclude_dirs}
            nf_dirs = [d for d in nf_dirs if d.resolve() not in excluded]
        if not nf_dirs:
            raise ValueError(f"No Neurofinder directories found under {data_dir}")

        self._videos: list[dict] = []
        self.index: list[tuple[int, int]] = []

        for nf_dir in nf_dirs:
            img_dir = nf_dir / "images"
            paths = sorted(img_dir.glob("*.tiff")) or sorted(img_dir.glob("*.tif"))
            if not paths:
                continue

            first = np.array(Image.open(str(paths[0])))
            orig_h, orig_w = first.shape[0], first.shape[1]
            mask = _build_nf_mask(nf_dir, orig_h, orig_w, img_size) if labeled else None

            v_idx = len(self._videos)
            self._videos.append({"paths": paths, "mask": mask})

            T = len(paths)
            if T < seq_len:
                self.index.append((v_idx, 0))
            else:
                for start in range(0, T - seq_len + 1, stride):
                    self.index.append((v_idx, start))

        if not self.index:
            raise ValueError(f"No image frames found under {data_dir}")

        if labeled_fraction < 1.0:
            rng = np.random.default_rng(seed)
            n = max(1, int(len(self.index) * labeled_fraction))
            keep = rng.choice(len(self.index), size=n, replace=False)
            self.index = [self.index[i] for i in sorted(keep)]

    def __len__(self) -> int:
        """Return the number of temporal clips across all loaded Neurofinder datasets."""
        return len(self.index)

    def _load_clip(self, paths: list, start: int) -> np.ndarray:
        """Lazily read seq_len frames from disk, normalize, resize, and zero-pad if short."""
        frames = []
        for p in paths[start : start + self.seq_len]:
            frame = np.array(Image.open(str(p))).astype(np.float32)
            if frame.ndim == 3:
                frame = frame[:, :, 0]
            frames.append(frame)
        clip = _normalize(np.stack(frames))
        if clip.shape[-1] != self.img_size or clip.shape[-2] != self.img_size:
            clip = _resize_frames(clip, self.img_size, cv2.INTER_LINEAR)
        if clip.shape[0] < self.seq_len:
            padded = np.zeros((self.seq_len, self.img_size, self.img_size), dtype=clip.dtype)
            padded[: clip.shape[0]] = clip
            clip = padded
        return clip

    def __getitem__(self, idx: int) -> dict:
        """Return a dict with 'video' and optionally 'mask' tensors for the given clip."""
        v_idx, start = self.index[idx]
        video = self._videos[v_idx]
        clip = self._load_clip(video["paths"], start)
        video_t = torch.from_numpy(clip).unsqueeze(0).float()

        mask = video["mask"]
        if mask is None or not self.labeled:
            return {"video": video_t}

        mask_seq = np.stack([mask] * self.seq_len)
        mask_t = (
            torch.from_numpy((mask_seq > 0).astype(np.float32))
            if self.binarize
            else torch.from_numpy(mask_seq.astype(np.int64))
        )
        return {"video": video_t, "mask": mask_t}
