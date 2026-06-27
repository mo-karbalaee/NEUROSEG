import argparse
from pathlib import Path
from typing import Any

from neuroseg.models.hypothesis import Hypothesis
from neuroseg.models.mode import Mode
from neuroseg.pipeline import run


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="neuroseg — neural segmentation pipeline for calcium imaging"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory of TIFF stacks (training data or inference input).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory (checkpoints for training, results for inference).",
    )

    hyp_group = parser.add_mutually_exclusive_group()
    hyp_group.add_argument("--H1", action="store_true", help="Run H1 experiment.")
    hyp_group.add_argument("--H2", action="store_true", help="Run H2 experiment.")
    hyp_group.add_argument("--H3", action="store_true", help="Run H3 experiment.")

    parser.add_argument(
        "--labeled-data",
        type=Path,
        default=None,
        metavar="DIR",
        help="(H1) Directory of labeled data for fine-tuning.",
    )
    parser.add_argument(
        "--labeled-fractions",
        nargs="+",
        type=float,
        default=None,
        metavar="F",
        help="(H1) Labeled fractions to evaluate, e.g. 0.01 0.05 0.1 1.0.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=100,
        metavar="N",
        help="(H1) Number of self-supervised pretraining epochs.",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=50,
        metavar="N",
        help="(H1/H2) Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=None,
        metavar="DIR",
        help="(H2) Source-organism data directory. Organism inferred from Neurofinder directory names.",
    )
    parser.add_argument(
        "--target-data",
        type=Path,
        default=None,
        metavar="DIR",
        help="(H2) Target-organism labeled data directory. Organism inferred from Neurofinder directory names.",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=None,
        metavar="FILE",
        help="(H3) Path to pretrained JEPA checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="YAML config file. Sets any H1Config field and hypothesis-specific keys. "
             "CLI flags take precedence over values in the file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="FILE",
        help="(inference) Use this checkpoint file directly, skipping the interactive picker.",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    mode = Mode.TRAINING if args.mode == "train" else Mode.INFERENCE

    hypothesis = None
    if mode == Mode.TRAINING:
        if not (args.H1 or args.H2 or args.H3):
            raise SystemExit("--mode train requires exactly one of --H1, --H2, --H3.")
        if args.H1:
            hypothesis = Hypothesis.H1
        elif args.H2:
            hypothesis = Hypothesis.H2
        else:
            hypothesis = Hypothesis.H3

    checkpoint_path = None
    if mode == Mode.INFERENCE:
        if args.checkpoint:
            checkpoint_path = str(args.checkpoint)
        else:
            from neuroseg.cli import pick_checkpoint
            selected = pick_checkpoint(args.output)
            checkpoint_path = str(selected) if selected else None

    config: dict = _load_yaml(args.config) if args.config else {}

    config.setdefault("pretrain_epochs", args.pretrain_epochs)
    config.setdefault("finetune_epochs", args.finetune_epochs)
    if args.labeled_data is not None:
        config["labeled_data_dir"] = str(args.labeled_data)
    if args.labeled_fractions is not None:
        config["labeled_fractions"] = args.labeled_fractions
    if args.source_data is not None:
        config["source_data_dir"] = str(args.source_data)
    if args.target_data is not None:
        config["target_data_dir"] = str(args.target_data)
    if args.pretrained_ckpt is not None:
        config["pretrained_ckpt"] = str(args.pretrained_ckpt)

    run(
        data_dir=args.data,
        output_dir=args.output,
        mode=mode,
        hypothesis=hypothesis,
        checkpoint_path=checkpoint_path,
        config=config,
    )


if __name__ == "__main__":
    main()
