from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neuroseg.logger import RunLogger
from neuroseg.models.state import State
from neuroseg.trainers.dataset import find_neurofinder_dirs
from neuroseg.trainers.h1_trainer import (
    H1Config,
    build_config,
    finetune,
    pretrain,
    setup_seed,
    _make_labeled_dataset,
    _validate_finetune,
)

_NF_ORGANISM_MAP = {
    "00": "mouse.visual_cortex",
    "01": "mouse.visual_cortex",
    "02": "mouse.visual_cortex",
    "03": "mouse.visual_cortex",
    "04": "zebrafish",
    "10": "mouse.hippocampus",
}


def _infer_organism(data_dir: str) -> str:
    """Infer the organism label from Neurofinder dataset IDs found under data_dir."""
    nf_dirs = find_neurofinder_dirs(data_dir)
    if not nf_dirs:
        return Path(data_dir).name
    dataset_ids = set()
    for d in nf_dirs:
        parts = d.name.split(".")
        if len(parts) >= 2:
            dataset_ids.add(parts[1])
    organisms = {_NF_ORGANISM_MAP.get(did, did) for did in dataset_ids}
    return "+".join(sorted(organisms))


def _build_h2_config(state: State) -> H1Config:
    """Build the H2 training config (an H1Config populated from config file and CLI)."""
    return build_config(state)


@torch.inference_mode()
def evaluate_on_target(checkpoint_path: str, target_dir: str, cfg: H1Config, device: torch.device) -> dict:
    """Load a source-trained segmentation model and evaluate it zero-shot on the target organism."""
    from neuroseg.checkpoint import load_compound_checkpoint
    from neuroseg.trainers.jepa import build_jepa, build_seg_head

    payload = load_compound_checkpoint(Path(checkpoint_path))
    arch = payload["arch"]
    jepa = build_jepa(arch, device)
    jepa.load_state_dict(payload["jepa"], strict=False)
    seg_head = build_seg_head(arch["dstc"], arch.get("seg_head_hidden", 16)).to(device)
    seg_head.load_state_dict(payload["seg_head"])

    dataset = _make_labeled_dataset(target_dir, cfg, fraction=1.0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    dice_score, miou_score = _validate_finetune(loader, jepa, seg_head, device)
    return {"dice": dice_score, "miou": miou_score}


def run_h2(state: State):
    """
    H2 — Cross-organism transfer.

    Protocol
    --------
    1. Train two segmentation models entirely on the SOURCE organism:
         - JEPA-pretrained: self-supervised JEPA pretraining on the source, then
           supervised segmentation training on the source labels.
         - Supervised baseline: supervised segmentation from random init on the source labels.
    2. Evaluate BOTH models zero-shot on the TARGET organism — no target training.
    3. Compare target Dice / mIoU, and the source→target drop, between the two models.
       Higher target score / smaller drop = better cross-organism generalization.

    Any two Neurofinder datasets can be used as source/target — the organism label
    is inferred automatically from the Neurofinder directory names. Each of
    source_data_dir and target_data_dir may be a single Neurofinder directory or a
    parent directory of several.

    Required config keys
    --------------------
    source_data_dir : str  — source-organism directory (models are trained here)
    target_data_dir : str  — target-organism directory (models are evaluated here, zero-shot)

    Logging
    -------
    A per-model summary row is appended to <output>/logs/runs.csv with hypothesis=H2,
    mode in {finetune, supervised_baseline}, source test scores in test_dice/test_miou,
    and zero-shot target scores in target_dice/target_miou.
    """
    cfg = _build_h2_config(state)
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(state["output_dir"])
    extra = state.get("config", {})

    source_dir = extra.get("source_data_dir", "")
    target_dir = extra.get("target_data_dir", "")

    if not source_dir:
        raise ValueError("H2 requires 'source_data_dir' in config (via --source-data).")
    if not target_dir:
        raise ValueError("H2 requires 'target_data_dir' in config (via --target-data).")

    source_organism = _infer_organism(source_dir)
    target_organism = _infer_organism(target_dir)

    print(
        f"[H2] device={device} | train on {source_organism} ({source_dir}) "
        f"| eval on {target_organism} ({target_dir})"
    )

    log_path = output_dir / "logs" / "runs.csv"

    pretrained_path = pretrain(
        source_dir, cfg, output_dir, device,
        model_name=f"jepa_pretrained_h2_{source_organism}",
        log_path=log_path, hypothesis="H2",
    )

    jepa_res = finetune(
        pretrained_path, source_dir, fraction=1.0, cfg=cfg,
        output_dir=output_dir, device=device, mode="finetune",
        model_name=f"jepa_h2_jepa_{source_organism}",
        log_path=log_path, hypothesis="H2",
    )
    supervised_res = finetune(
        None, source_dir, fraction=1.0, cfg=cfg,
        output_dir=output_dir, device=device, mode="supervised_baseline",
        model_name=f"jepa_h2_supervised_{source_organism}",
        log_path=log_path, hypothesis="H2",
    )

    for mode, res in [("finetune", jepa_res), ("supervised_baseline", supervised_res)]:
        if not res or "checkpoint" not in res:
            continue
        target_metrics = evaluate_on_target(res["checkpoint"], target_dir, cfg, device)
        logger = RunLogger(log_path, hypothesis="H2", mode=mode, model_name=f"h2_transfer_{mode}")
        logger.log(
            test_dice=res["dice"], test_miou=res["miou"],
            target_dice=target_metrics["dice"], target_miou=target_metrics["miou"],
            checkpoint=res["checkpoint"],
        )
        print(
            f"[H2/{mode}] source dice={res['dice']:.4f} → target dice={target_metrics['dice']:.4f} "
            f"(drop {res['dice'] - target_metrics['dice']:+.4f})"
        )

    print("[H2] Done.")

    from neuroseg.plots import plot_h2_target, plot_h2_drop
    figures_dir = output_dir / "figures"
    plot_h2_target(log_path, figures_dir)
    plot_h2_drop(log_path, figures_dir)
