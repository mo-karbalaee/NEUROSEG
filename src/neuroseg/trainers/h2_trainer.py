import json
from pathlib import Path

import torch

from neuroseg.models.state import State
from neuroseg.trainers.h1_trainer import (
    build_config,
    finetune,
    pretrain,
    setup_seed,
)


def run_h2(state: State):
    """
    H2 — Cross-species transfer (limited fine-tuning).

    Protocol
    --------
    1. Obtain a JEPA encoder self-supervised on the non-mouse SOURCE (Drosophila +
       zebrafish): either pretrain on ``--source-data``, or reuse an already-trained
       checkpoint via ``--pretrained-ckpt`` (its architecture is read from the JSON
       sidecar so it loads correctly).
    2. Fine-tune on the labeled mouse TARGET (``--target-data``) at each labeled
       fraction, using the SAME working pipeline as H1 (``finetune``: encoder + seg
       head, correct masks, fixed 80/20 test split, min-max normalized input):
       cross-species-pretrained init (mode ``finetune``) vs from-scratch
       (mode ``supervised_baseline``).
    3. Higher target Dice for the pretrained arm at a given fraction = the cross-species
       representation transfers to mouse. The gap is largest at small fractions
       ("limited fine-tuning"), where a good initialization matters most.

    Leakage
    -------
    Clean by construction: the encoder is pretrained on a different species than the
    mouse target, so it never sees the mouse test set. ``finetune`` additionally holds
    out a fixed 20% of the target recording for testing (identical across fractions).

    Required config keys
    --------------------
    target_data_dir : str  — labeled mouse Neurofinder directory (fine-tune + test target)
    source_data_dir : str  — non-mouse pretraining video directory (TIFF and/or CZI);
                             required unless pretrained_ckpt is given
    pretrained_ckpt : str  — (optional) path to an already-trained cross-species JEPA
                             checkpoint; skips pretraining and reads arch from its sidecar
    labeled_fractions : list[float] — target label fractions to fine-tune at

    Logging
    -------
    Rows are appended to <output>/logs/runs.csv with hypothesis=H2 and mode in
    {finetune, supervised_baseline}; held-out target scores are in test_dice/test_miou.
    """
    cfg = build_config(state)
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(state["output_dir"])
    extra = state.get("config", {})

    source_dir = extra.get("source_data_dir", "")
    target_dir = extra.get("target_data_dir", "")
    pretrained_ckpt = extra.get("pretrained_ckpt")
    fractions = extra.get("labeled_fractions", cfg.labeled_fractions)
    log_path = output_dir / "logs" / "runs.csv"

    if not target_dir:
        raise ValueError("H2 requires 'target_data_dir' (via --target-data): the labeled mouse target.")
    if not pretrained_ckpt and not source_dir:
        raise ValueError(
            "H2 requires 'source_data_dir' (via --source-data) to pretrain, or "
            "'pretrained_ckpt' (via --pretrained-ckpt) to reuse an already-trained JEPA checkpoint."
        )

    if pretrained_ckpt:
        sidecar = Path(pretrained_ckpt).with_suffix(".json")
        if sidecar.exists():
            for k, v in json.loads(sidecar.read_text()).get("arch", {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        pretrained_path = str(pretrained_ckpt)
        print(
            f"[H2] device={device} | reusing cross-species encoder (skip pretraining): "
            f"{pretrained_path} | target={target_dir} | dstc={cfg.dstc}"
        )
    else:
        print(
            f"[H2] device={device} | pretrain(source)={source_dir} | target={target_dir} "
            f"| dstc={cfg.dstc} | augment={cfg.pretrain_augment}"
        )
        pretrained_path = pretrain(
            source_dir, cfg, output_dir, device,
            model_name="jepa_pretrained_h2", log_path=log_path, hypothesis="H2",
        )
        if pretrained_path is None:
            raise ValueError("H2 pretraining produced no checkpoint — check --source-data.")

    for fraction in fractions:
        finetune(
            pretrained_path, target_dir, fraction, cfg, output_dir, device,
            mode="finetune", model_name=f"jepa_h2_finetune_f{int(fraction * 100)}",
            log_path=log_path, hypothesis="H2",
        )
        finetune(
            None, target_dir, fraction, cfg, output_dir, device,
            mode="supervised_baseline", model_name=f"jepa_h2_supervised_f{int(fraction * 100)}",
            log_path=log_path, hypothesis="H2",
        )

    print("[H2] Done.")

    from neuroseg.plots import plot_h1_dice
    plot_h1_dice(
        log_path, output_dir / "figures",
        prefix="h2", title="H2 — Cross-species Transfer (source→mouse)",
        pretrained_label="Cross-species Pretrained",
    )
