from pathlib import Path

import torch

from neuroseg.models.state import State
from neuroseg.trainers.h1_trainer import H1Config, build_config, finetune, pretrain, setup_seed


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
    1. Pretrain JEPA on zebrafish calcium imaging (unlabeled).
    2. Fine-tune on Drosophila with a limited epoch budget.
    3. Train a supervised baseline from scratch on Drosophila with the same budget.
    4. Compare Dice / mIoU drop between source and target organism.

    Required config keys
    --------------------
    zebrafish_data_dir  : str  — directory of unlabeled zebrafish TIFF stacks
    drosophila_data_dir : str  — directory of labeled Drosophila data
                                 (LabeledTIFFDataset layout: sample/video.tif + mask.tif)
    finetune_budget     : int  — fine-tuning epochs on target organism (default 10)

    MLflow tags
    -----------
    hypothesis=H2, source_organism=zebrafish, target_organism=drosophila,
    mode={pretrain | finetune | supervised_baseline}
    """
    cfg = _build_h2_config(state)
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(state["output_dir"])
    extra = state.get("config", {})

    zebrafish_dir = extra.get("zebrafish_data_dir", "")
    drosophila_dir = extra.get("drosophila_data_dir", "")

    if not zebrafish_dir:
        raise ValueError("H2 requires 'zebrafish_data_dir' in config (via --zebrafish-data).")
    if not drosophila_dir:
        raise ValueError("H2 requires 'drosophila_data_dir' in config (via --drosophila-data).")

    zebrafish_files = [
        str(p) for p in sorted(Path(zebrafish_dir).iterdir()) if p.is_file()
    ]
    print(
        f"[H2] device={device} | zebrafish files={len(zebrafish_files)} "
        f"| drosophila dir={drosophila_dir}"
    )

    organism_tags = {
        "hypothesis": "H2",
        "source_organism": "zebrafish",
        "target_organism": "drosophila",
    }

    pretrained_path = pretrain(
        zebrafish_files,
        cfg,
        output_dir,
        device,
        mlflow_experiment="neuroseg-H2-pretrain",
        base_tags={**organism_tags, "model_name": "jepa_pretrained_h2_zebrafish"},
    )

    finetune(
        pretrained_path,
        drosophila_dir,
        fraction=1.0,
        cfg=cfg,
        output_dir=output_dir,
        device=device,
        mode="finetune",
        mlflow_experiment="neuroseg-H2-finetune",
        base_tags={**organism_tags, "model_name": "jepa_h2_finetune_drosophila"},
    )

    finetune(
        None,
        drosophila_dir,
        fraction=1.0,
        cfg=cfg,
        output_dir=output_dir,
        device=device,
        mode="supervised_baseline",
        mlflow_experiment="neuroseg-H2-finetune",
        base_tags={**organism_tags, "model_name": "jepa_h2_supervised_drosophila"},
    )

    print("[H2] Done.")
