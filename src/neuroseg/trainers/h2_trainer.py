from pathlib import Path

import torch

from neuroseg.models.state import State
from neuroseg.trainers.dataset import find_neurofinder_dirs
from neuroseg.trainers.h1_trainer import H1Config, build_config, finetune, pretrain, setup_seed

_NF_ORGANISM_MAP = {
    "00": "mouse.visual_cortex",
    "01": "mouse.visual_cortex",
    "02": "mouse.visual_cortex",
    "03": "mouse.visual_cortex",
    "04": "zebrafish",
    "10": "mouse.hippocampus",
}


def _infer_organism(data_dir: str) -> str:
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
    cfg = build_config(state)
    extra = state.get("config", {})
    cfg.pretrain_epochs = extra.get("pretrain_epochs", cfg.pretrain_epochs)
    cfg.finetune_epochs = extra.get("finetune_budget", 10)
    return cfg


def run_h2(state: State):
    """
    H2 — Cross-organism transfer.

    Protocol
    --------
    1. Pretrain JEPA on source-organism calcium imaging (unlabeled).
    2. Fine-tune on target-organism data with a limited epoch budget.
    3. Train a supervised baseline from scratch on target-organism data with the same budget.
    4. Compare Dice / mIoU drop from source to target.

    Any two Neurofinder datasets can be used as source/target — the organism
    label is inferred automatically from the Neurofinder directory names.

    Both source_data_dir and target_data_dir may be either:
    - A single Neurofinder directory (contains images/)
    - A parent directory whose subdirectories are Neurofinder datasets

    Required config keys
    --------------------
    source_data_dir : str  — source-organism directory (e.g. Neurofinder 04 for zebrafish)
    target_data_dir : str  — target-organism directory (e.g. Neurofinder 00–03 for mouse)
    finetune_budget : int  — fine-tuning epochs on target organism (default 10)

    MLflow tags
    -----------
    hypothesis=H2, source_organism={inferred}, target_organism={inferred},
    mode={pretrain | finetune | supervised_baseline}
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
        f"[H2] device={device} | source={source_organism} ({source_dir}) "
        f"| target={target_organism} ({target_dir})"
    )

    organism_tags = {
        "hypothesis": "H2",
        "source_organism": source_organism,
        "target_organism": target_organism,
    }

    pretrained_path = pretrain(
        source_dir,
        cfg,
        output_dir,
        device,
        mlflow_experiment="neuroseg-H2-pretrain",
        base_tags={**organism_tags, "model_name": f"jepa_pretrained_h2_{source_organism}"},
    )

    finetune(
        pretrained_path,
        target_dir,
        fraction=1.0,
        cfg=cfg,
        output_dir=output_dir,
        device=device,
        mode="finetune",
        mlflow_experiment="neuroseg-H2-finetune",
        base_tags={**organism_tags, "model_name": f"jepa_h2_finetune_{target_organism}"},
    )

    finetune(
        None,
        target_dir,
        fraction=1.0,
        cfg=cfg,
        output_dir=output_dir,
        device=device,
        mode="supervised_baseline",
        mlflow_experiment="neuroseg-H2-finetune",
        base_tags={**organism_tags, "model_name": f"jepa_h2_supervised_{target_organism}"},
    )

    print("[H2] Done.")
