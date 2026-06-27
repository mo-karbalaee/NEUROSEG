from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


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


def _resize_clip(clip: np.ndarray, img_size: int) -> np.ndarray:
    T = clip.shape[0]
    return np.stack([
        cv2.resize(clip[t], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        for t in range(T)
    ])


class TIFFVideoDataset(Dataset):
    """Unlabeled dataset for JEPA self-supervised pretraining on TIFF stacks."""

    def __init__(self, file_paths: list[str], seq_len: int, img_size: int):
        self.seq_len = seq_len
        self.img_size = img_size
        self.clips: list[np.ndarray] = []

        for path in file_paths:
            video = _normalize(_load_tiff(path))
            T = video.shape[0]
            for start in range(0, T - seq_len, seq_len):
                self.clips.append(video[start : start + seq_len])

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        if clip.shape[-1] != self.img_size or clip.shape[-2] != self.img_size:
            clip = _resize_clip(clip, self.img_size)
        video = torch.from_numpy(clip).unsqueeze(0).float()
        return {"video": video}


class LabeledTIFFDataset(Dataset):
    """
    Labeled dataset for H1 fine-tuning and supervised baseline.

    Expected directory layout:
        data_dir/
            sample_001/
                video.tif   — multi-frame TIFF stack (T, H, W)
                mask.tif    — single-frame binary mask (H, W) or (T, H, W)
            sample_002/
                ...

    Each mask is broadcast to match the video frame count.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        img_size: int,
        labeled_fraction: float = 1.0,
        seed: int = 0,
    ):
        self.seq_len = seq_len
        self.img_size = img_size
        samples = sorted(Path(data_dir).iterdir())
        rng = np.random.default_rng(seed)
        n = max(1, int(len(samples) * labeled_fraction))
        idx = rng.choice(len(samples), size=n, replace=False)
        self.samples = [samples[i] for i in sorted(idx)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]
        video = _normalize(_load_tiff(str(sample_dir / "video.tif")))
        mask = _load_tiff(str(sample_dir / "mask.tif"))

        if mask.ndim == 2:
            mask = np.stack([mask] * video.shape[0])

        if video.shape[-1] != self.img_size or video.shape[-2] != self.img_size:
            video = _resize_clip(video, self.img_size)
            mask = _resize_clip(mask, self.img_size)

        T = min(video.shape[0], self.seq_len)
        video_t = torch.from_numpy(video[:T]).unsqueeze(0).float()
        mask_t = torch.from_numpy((mask[:T] > 0).astype(np.float32))
        return {"video": video_t, "mask": mask_t}
