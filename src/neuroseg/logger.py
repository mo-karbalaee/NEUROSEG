import csv
import uuid
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "timestamp", "run_id", "hypothesis", "mode", "model_name",
    "labeled_fraction", "epoch", "train_loss", "train_recon_loss",
    "val_jepa_loss", "val_recon_loss",
    "val_dice", "val_miou", "test_dice", "test_miou",
    "target_dice", "target_miou",
    "within_sim", "between_sim", "gap", "checkpoint",
]


class RunLogger:
    def __init__(
        self,
        log_path: Path,
        hypothesis: str,
        mode: str,
        model_name: str,
        labeled_fraction: float | None = None,
    ):
        """Initialize a logger for one training run and assign it a unique run_id."""
        self.run_id = uuid.uuid4().hex[:8]
        self._meta = {
            "hypothesis": hypothesis,
            "mode": mode,
            "model_name": model_name,
            "labeled_fraction": "" if labeled_fraction is None else labeled_fraction,
        }
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, epoch: int | None = None, **metrics):
        """Append one row with the current metrics and run metadata to the CSV log."""
        row = {col: "" for col in COLUMNS}
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        row["run_id"] = self.run_id
        row.update(self._meta)
        if epoch is not None:
            row["epoch"] = epoch
        for k, v in metrics.items():
            if k in row:
                row[k] = f"{v:.6f}" if isinstance(v, float) else v
        write_header = not self._path.exists() or self._path.stat().st_size == 0
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
