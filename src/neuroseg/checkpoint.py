import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    model,
    model_name: str,
    run_id: str,
    output_dir: Path,
    metadata: Optional[dict] = None,
    arch: Optional[dict] = None,
) -> Path:
    """Save a single model's state dict and write a JSON sidecar with metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{model_name}_{run_id}.pt"
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")

    torch.save(model.state_dict(), path)
    _write_sidecar(path, model_name, run_id, metadata, arch=arch)
    return path


def save_compound_checkpoint(
    models: dict,
    arch: dict,
    model_name: str,
    run_id: str,
    output_dir: Path,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save a compound checkpoint containing multiple model state dicts
    and the architecture config needed to reconstruct them.

    `models` is a mapping from component name → nn.Module,
    e.g. {"jepa": jepa_model, "seg_head": seg_head_model}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{model_name}_{run_id}.pt"
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")

    payload = {
        "type": "neuroseg_jepa_v1",
        "arch": arch,
        **{k: v.state_dict() for k, v in models.items()},
    }
    torch.save(payload, path)
    sidecar_meta = {"compound": True, **(metadata or {})}
    _write_sidecar(path, model_name, run_id, sidecar_meta)
    return path


def save_latest_checkpoint(
    model,
    model_name: str,
    run_id: str,
    output_dir: Path,
    arch: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> Path:
    """Overwrite a rolling '<name>_<run_id>_latest.pt' checkpoint for crash recovery during long runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{model_name}_{run_id}_latest.pt"
    torch.save(model.state_dict(), path)

    meta = {"model_name": model_name, "run_id": run_id, "date": datetime.now().isoformat()}
    if metadata:
        meta.update(metadata)
    if arch:
        meta["arch"] = arch
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def load_compound_checkpoint(path: Path) -> dict:
    """Load and validate a compound checkpoint saved by save_compound_checkpoint."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("type") != "neuroseg_jepa_v1":
        raise ValueError(
            f"Expected checkpoint type 'neuroseg_jepa_v1', got '{payload.get('type')}'. "
            "This checkpoint was not saved by save_compound_checkpoint."
        )
    return payload


def list_checkpoints(output_dir: Path) -> list[dict]:
    """Return only compound (inference-capable) checkpoints — those that contain
    both an encoder and a seg_head and can therefore produce segmentation masks."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    results = []
    for cp in sorted(output_dir.glob("*.pt")):
        meta_path = cp.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if not meta.get("compound"):
                continue
        else:
            continue
        meta["path"] = cp
        results.append(meta)
    return results


def _write_sidecar(path: Path, model_name: str, run_id: str, metadata: Optional[dict], arch: Optional[dict] = None):
    """Write a JSON metadata sidecar file next to the checkpoint."""
    meta = {
        "model_name": model_name,
        "run_id": run_id,
        "date": datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)
    if arch:
        meta["arch"] = arch
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
