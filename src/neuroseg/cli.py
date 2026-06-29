from pathlib import Path
from typing import Optional

from neuroseg.checkpoint import list_checkpoints


def pick_checkpoint(output_dir: Path) -> Optional[Path]:
    """Interactively list compound checkpoints and return the user's selection."""
    checkpoints = list_checkpoints(Path(output_dir))

    if not checkpoints:
        print("No checkpoints found. Running with Cellpose baseline.")
        return None

    print("\nAvailable checkpoints:")
    for i, meta in enumerate(checkpoints):
        dice_str = f"Dice={meta['dice']:.4f}" if meta.get("dice") is not None else "Dice=N/A"
        miou_str = f"mIoU={meta['miou']:.4f}" if meta.get("miou") is not None else "mIoU=N/A"
        date_str = (meta.get("date") or "unknown")[:10]
        name = meta.get("model_name", meta["path"].stem)
        hyp = meta.get("hypothesis", "")
        print(f"  [{i + 1}] {name}  {hyp}  {date_str}  {dice_str}  {miou_str}")
    print("  [0] Skip — use Cellpose baseline")

    while True:
        try:
            raw = input(f"\nSelect checkpoint [0-{len(checkpoints)}]: ").strip()
            choice = int(raw)
            if 0 <= choice <= len(checkpoints):
                return checkpoints[choice - 1]["path"] if choice > 0 else None
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        print("Invalid choice, try again.")
