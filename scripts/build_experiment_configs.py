import argparse
import csv
from pathlib import Path

import yaml


def _parse_hparams(s: str) -> dict:
    """Parse a 'key=value;key=value' hyperparameter string into a dict."""
    out = {}
    for part in s.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _row_to_config(row: dict) -> dict:
    """Build the per-run config dict for one experiments.csv row, linked by its UID."""
    cfg = {
        "uid": row["uid"],
        "date": row["date"],
        "hypothesis": row["hypothesis"],
        "model": row["model"],
        "dataset": row["dataset"],
        "init": row["init"],
        "seed": int(row["seed"]) if row["seed"] else None,
        "labeled_fraction": float(row["labeled_fraction"]) if row["labeled_fraction"] else None,
        "hyperparameters": _parse_hparams(row["hyperparameters"]),
        "tag": row["tag"],
    }
    return cfg


def main() -> None:
    """Generate one config_<uid>.yaml per row of experiments.csv, linked by UID."""
    ap = argparse.ArgumentParser(description="Generate per-run config files from experiments.csv.")
    ap.add_argument("--csv", type=Path, default=Path("experiments/experiments.csv"))
    ap.add_argument("--out", type=Path, default=Path("experiments/configs"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.csv, newline="")))
    for row in rows:
        cfg = _row_to_config(row)
        path = args.out / f"config_{row['uid']}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {len(rows)} config files to {args.out}")


if __name__ == "__main__":
    main()
