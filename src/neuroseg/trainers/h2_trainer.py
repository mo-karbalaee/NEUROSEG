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


def probe_on_target(pretrained_ckpt, target_dir: str, cfg: H1Config, device: torch.device,
                    fraction: float, probe_epochs: int = 50, lr: float = 1e-3) -> dict:
    """
    Linear-probe transfer to the target (mouse).

    Freeze a JEPA encoder (loaded from a pretrained checkpoint, or random-init if None),
    train ONLY a fresh segmentation head on a `fraction` of target labels, and evaluate
    mean Dice / mIoU on a held-out target split. The encoder never trains on the target,
    so this measures how well the pretrained representation transfers.
    """
    import numpy as np
    import torch.nn as nn
    from torch.optim import Adam

    from neuroseg.metrics import dice as dice_fn, miou as miou_fn
    from neuroseg.trainers.jepa import build_jepa, build_seg_head

    jepa = build_jepa(cfg.arch_dict(), device)
    if pretrained_ckpt is not None:
        state_dict = torch.load(str(pretrained_ckpt), map_location=device, weights_only=True)
        jepa.load_state_dict(state_dict, strict=False)
    encoder = jepa.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    dataset = _make_labeled_dataset(target_dir, cfg, fraction=1.0)
    n = len(dataset)
    if n < 2:
        return {"dice": 0.0, "miou": 0.0}
    order = np.random.default_rng(cfg.seed).permutation(n)
    n_test = max(1, int(cfg.test_split * n))
    test_idx = order[:n_test]
    pool = order[n_test:]
    k = max(1, int(fraction * len(pool)))
    train_idx = pool[:k]

    @torch.inference_mode()
    def features(indices):
        feats, masks = [], []
        for i in indices:
            s = dataset[int(i)]
            x = s["video"].unsqueeze(0).to(device)
            feats.append(encoder(x).mean(dim=2).squeeze(0))
            masks.append((s["mask"][0] > 0).float().unsqueeze(0).to(device))
        return feats, masks

    ftr, mtr = features(train_idx)
    fte, mte = features(test_idx)

    head = build_seg_head(cfg.dstc, cfg.seg_head_hidden).to(device)
    optimizer = Adam(head.parameters(), lr=lr)
    criterion = nn.BCELoss()
    X, Y = torch.stack(ftr), torch.stack(mtr)
    bs = max(1, cfg.batch_size)
    for _ in range(probe_epochs):
        perm = torch.randperm(len(X))
        for j in range(0, len(X), bs):
            b = perm[j:j + bs]
            loss = criterion(head(X[b]), Y[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    head.eval()
    dice_scores, miou_scores = [], []
    with torch.inference_mode():
        for f, m in zip(fte, mte):
            pred = head(f.unsqueeze(0)).squeeze().cpu().numpy()
            pb = (pred > 0.5).astype(np.uint8)
            gt = m.squeeze().cpu().numpy().astype(np.uint8)
            dice_scores.append(dice_fn(pb, gt))
            miou_scores.append(miou_fn(pb, gt, num_classes=2))
    return {"dice": float(np.mean(dice_scores)), "miou": float(np.mean(miou_scores))}


def run_h2(state: State):
    """
    H2 — Cross-species transfer.

    Protocol
    --------
    1. Pretrain a JEPA encoder self-supervised on the SOURCE (non-mouse) video
       (`--source-data`) — e.g. Drosophila larvae + zebrafish calcium imaging. No labels needed.
    2. Linear-probe on the TARGET mouse subsection (`--target-data`, a labeled Neurofinder set):
       freeze the encoder and train only a fresh segmentation head on a small `probe_fraction`
       of mouse labels, then evaluate on held-out mouse.
    3. Compare the pretrained encoder against a from-scratch (random-init) encoder under the
       same probe. Higher target Dice for the pretrained encoder = the source representation
       transfers to mouse.

    Required config keys
    --------------------
    source_data_dir : str  — non-mouse pretraining video directory (TIFF and/or CZI)
    target_data_dir : str  — labeled mouse Neurofinder directory (probe target)
    probe_fraction  : float — fraction of target labels used to train the probe head (default 0.1)
    pretrained_ckpt : str  — (optional, via --pretrained-ckpt) path to an already-trained JEPA
                             checkpoint; when given, pretraining is skipped and its architecture
                             is read from the checkpoint's JSON sidecar. source_data_dir is then
                             not required.

    Logging
    -------
    One row per mode is appended to <output>/logs/runs.csv with hypothesis=H2,
    mode in {pretrained, from_scratch}, and held-out target scores in target_dice/target_miou.
    """
    cfg = _build_h2_config(state)
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(state["output_dir"])
    extra = state.get("config", {})

    source_dir = extra.get("source_data_dir", "")
    target_dir = extra.get("target_data_dir", "")
    pretrained_ckpt = extra.get("pretrained_ckpt")

    if not target_dir:
        raise ValueError("H2 requires 'target_data_dir' (via --target-data): the labeled mouse target.")
    if not pretrained_ckpt and not source_dir:
        raise ValueError(
            "H2 requires 'source_data_dir' (via --source-data) to pretrain, or "
            "'pretrained_ckpt' (via --pretrained-ckpt) to reuse an already-trained JEPA checkpoint."
        )

    probe_fraction = float(extra.get("probe_fraction", 0.1))
    log_path = output_dir / "logs" / "runs.csv"

    if pretrained_ckpt:
        import json
        sidecar = Path(pretrained_ckpt).with_suffix(".json")
        if sidecar.exists():
            for k, v in json.loads(sidecar.read_text()).get("arch", {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        pretrained_path = str(pretrained_ckpt)
        print(
            f"[H2] device={device} | reusing pretrained checkpoint (skipping pretraining): "
            f"{pretrained_path} | probe(target)={target_dir} | probe_fraction={probe_fraction}"
        )
    else:
        print(
            f"[H2] device={device} | pretrain(source)={source_dir} "
            f"| probe(target)={target_dir} | probe_fraction={probe_fraction}"
        )
        pretrained_path = pretrain(
            source_dir, cfg, output_dir, device,
            model_name="jepa_pretrained_h2", log_path=log_path, hypothesis="H2",
        )

    for mode, ckpt in [("pretrained", pretrained_path), ("from_scratch", None)]:
        metrics = probe_on_target(ckpt, target_dir, cfg, device, probe_fraction)
        RunLogger(log_path, hypothesis="H2", mode=mode, model_name=f"h2_probe_{mode}").log(
            labeled_fraction=probe_fraction,
            target_dice=metrics["dice"], target_miou=metrics["miou"],
        )
        print(f"[H2/probe {mode}] target Dice={metrics['dice']:.4f}  mIoU={metrics['miou']:.4f}")

    print("[H2] Done.")

    from neuroseg.plots import plot_h2_probe
    plot_h2_probe(log_path, output_dir / "figures")
