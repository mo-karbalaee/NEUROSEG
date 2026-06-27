import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional


def save_checkpoint(
    model,
    model_name: str,
    run_id: str,
    output_dir: Path,
    metadata: Optional[dict] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{model_name}_{run_id}.pt"
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")

    torch.save(model.state_dict(), path)

    meta = {
        "model_name": model_name,
        "run_id": run_id,
        "date": datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)

    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    return path


def list_checkpoints(output_dir: Path) -> list[dict]:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    results = []
    for cp in sorted(output_dir.glob("*.pt")):
        meta_path = cp.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {"model_name": cp.stem, "date": None, "dice": None, "miou": None}
        meta["path"] = cp
        results.append(meta)
    return results
