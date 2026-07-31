import argparse
import csv
from pathlib import Path


def _final_rows(log_path: Path) -> list:
    """Return the last row per (hypothesis, mode, fraction, model) in one run log."""
    if not log_path.exists():
        return []
    best = {}
    for r in csv.DictReader(open(log_path, newline="")):
        key = (r.get("hypothesis"), r.get("mode"), r.get("labeled_fraction"), r.get("model_name"))
        best[key] = r
    return list(best.values())


def _fmt(r: dict) -> str:
    """Format one run's final metrics into a compact single line."""
    cols = ["test_dice", "test_miou", "within_sim", "between_sim", "gap"]
    vals = [f"{c}={r[c]}" for c in cols if r.get(c, "") not in ("", None)]
    frac = r.get("labeled_fraction", "") or "-"
    return f"  {r.get('mode'):<20} frac={frac:<5} {r.get('model_name','')}: " + "  ".join(vals)


def main() -> None:
    """Print an aggregated summary of every training run found under an output directory."""
    ap = argparse.ArgumentParser(description="Aggregate NEUROSEG experiment logs into one summary.")
    ap.add_argument("--output", type=Path, default=Path("output"))
    args = ap.parse_args()

    logs = sorted(args.output.glob("*/logs/runs.csv"))
    if not logs:
        print(f"No runs.csv found under {args.output}")
        return
    for log_path in logs:
        rows = [r for r in _final_rows(log_path) if r.get("epoch", "") == "" or r.get("test_dice")
                or r.get("gap")]
        print("=" * 70)
        print(log_path)
        for r in sorted(rows, key=lambda r: (r.get("mode", ""), str(r.get("labeled_fraction")))):
            line = _fmt(r)
            if line.strip().endswith(":"):
                continue
            print(line)


if __name__ == "__main__":
    main()
