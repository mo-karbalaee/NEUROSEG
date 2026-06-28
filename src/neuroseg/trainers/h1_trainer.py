import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from neuroseg.checkpoint import save_checkpoint, save_compound_checkpoint
from neuroseg.metrics import dice, miou
from neuroseg.models.state import State
from neuroseg.trainers.dataset import (
    LabeledTIFFDataset,
    NeurofinderDataset,
    TIFFVideoDataset,
    is_neurofinder_dir,
    find_neurofinder_dirs,
)
from neuroseg.trainers.jepa import (
    JEPA,
    JEPAProbe,
    ImageDecoder,
    build_jepa,
    build_seg_head,
)

LABELED_FRACTIONS = [0.01, 0.05, 0.10, 1.0]


@dataclass
class H1Config:
    seq_len: int = 10
    img_size: int = 128
    batch_size: int = 8
    num_workers: int = 2
    henc: int = 32
    hpre: int = 32
    dstc: int = 8
    dobs: int = 1
    seg_head_hidden: int = 16
    std_coeff: float = 10.0
    cov_coeff: float = 100.0
    lr: float = 1e-3
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    steps: int = 4
    seed: int = 1
    labeled_fractions: list[float] = field(default_factory=lambda: list(LABELED_FRACTIONS))
    val_split: float = 0.1

    def arch_dict(self) -> dict:
        return {
            "dobs": self.dobs,
            "henc": self.henc,
            "hpre": self.hpre,
            "dstc": self.dstc,
            "seg_head_hidden": self.seg_head_hidden,
            "img_size": self.img_size,
        }


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(state: State) -> H1Config:
    cfg = H1Config()
    for k, v in state.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _make_unlabeled_dataset(data_dir: str, cfg: H1Config) -> Dataset:
    if is_neurofinder_dir(data_dir) or find_neurofinder_dirs(data_dir):
        return NeurofinderDataset(data_dir, cfg.seq_len, cfg.img_size, labeled=False)
    file_paths = [str(p) for p in sorted(Path(data_dir).iterdir()) if p.is_file()]
    return TIFFVideoDataset(file_paths, cfg.seq_len, cfg.img_size)


def _make_labeled_dataset(
    data_dir: str,
    cfg: H1Config,
    fraction: float,
    binarize: bool = True,
) -> Dataset:
    if is_neurofinder_dir(data_dir) or find_neurofinder_dirs(data_dir):
        return NeurofinderDataset(
            data_dir, cfg.seq_len, cfg.img_size,
            labeled=True, labeled_fraction=fraction,
            seed=cfg.seed, binarize=binarize,
        )
    return LabeledTIFFDataset(
        data_dir, cfg.seq_len, cfg.img_size,
        labeled_fraction=fraction, seed=cfg.seed, binarize=binarize,
    )


def pretrain(
    data_dir: str,
    cfg: H1Config,
    output_dir: Path,
    device: torch.device,
    mlflow_experiment: str = "neuroseg-H1-pretrain",
    base_tags: Optional[dict] = None,
) -> Optional[Path]:
    dataset = _make_unlabeled_dataset(data_dir, cfg)
    if len(dataset) == 0:
        print("No clips found for pretraining — skipping.")
        return None

    n_val = max(1, int(len(dataset) * cfg.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    jepa = build_jepa(cfg.arch_dict(), device)
    decoder = ImageDecoder(cfg.dstc, cfg.dobs, hidden_dim=16)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)

    optimizer = Adam([
        {"params": jepa.parameters(), "lr": cfg.lr},
        {"params": pixel_decoder.head.parameters(), "lr": cfg.lr / 10},
    ])

    tags = {"hypothesis": "H1", "mode": "pretrain"}
    if base_tags:
        tags.update(base_tags)

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(tags=tags) as run:
        mlflow.log_params({
            "seq_len": cfg.seq_len, "img_size": cfg.img_size,
            "batch_size": cfg.batch_size, "henc": cfg.henc, "hpre": cfg.hpre,
            "dstc": cfg.dstc, "std_coeff": cfg.std_coeff, "cov_coeff": cfg.cov_coeff,
            "lr": cfg.lr, "pretrain_epochs": cfg.pretrain_epochs, "steps": cfg.steps,
        })

        jepa.train()
        pixel_decoder.train()

        for epoch in range(cfg.pretrain_epochs):
            epoch_jepa, epoch_recon, n_batches = 0.0, 0.0, 0

            pbar = tqdm(train_loader, desc=f"[pretrain] Epoch {epoch}")
            for batch in pbar:
                x = batch["video"].to(device)
                optimizer.zero_grad()

                _, (jepa_loss, *_) = jepa.unroll(
                    x, None, nsteps=cfg.steps, unroll_mode="parallel", compute_loss=True
                )
                recon_loss = pixel_decoder(x, x)
                (jepa_loss + recon_loss).backward()
                optimizer.step()

                epoch_jepa += jepa_loss.item()
                epoch_recon += recon_loss.item()
                n_batches += 1
                pbar.set_postfix(jepa=f"{jepa_loss.item():.4f}", recon=f"{recon_loss.item():.4f}")

            val_metrics = _validate_pretrain(val_loader, jepa, pixel_decoder, cfg, device)
            mlflow.log_metrics(
                {"train/jepa_loss": epoch_jepa / n_batches,
                 "train/recon_loss": epoch_recon / n_batches,
                 **val_metrics},
                step=epoch,
            )

        run_id = run.info.run_id
        model_name = tags.get("model_name", "jepa_pretrained_h1")
        checkpoint_path = save_checkpoint(
            jepa, model_name=model_name, run_id=run_id,
            output_dir=output_dir, metadata={**tags},
        )
        mlflow.log_artifact(str(checkpoint_path))
        print(f"Pretrained checkpoint: {checkpoint_path}")

    return checkpoint_path


@torch.inference_mode()
def _validate_pretrain(val_loader, jepa: JEPA, pixel_decoder, cfg: H1Config, device: torch.device) -> dict:
    jepa.eval()
    pixel_decoder.eval()
    val_jepa, val_recon = [], []

    for batch in val_loader:
        x = batch["video"].to(device)
        _, (jepa_loss, *_) = jepa.unroll(
            x, None, nsteps=cfg.steps, unroll_mode="parallel", compute_loss=True
        )
        recon_loss = pixel_decoder(x, x)
        val_jepa.append(jepa_loss.item())
        val_recon.append(recon_loss.item())

    jepa.train()
    pixel_decoder.train()
    if not val_jepa:
        return {"val/jepa_loss": 0.0, "val/recon_loss": 0.0}
    return {"val/jepa_loss": float(np.mean(val_jepa)), "val/recon_loss": float(np.mean(val_recon))}


def finetune(
    pretrained_checkpoint: Optional[Path],
    labeled_data_dir: str,
    fraction: float,
    cfg: H1Config,
    output_dir: Path,
    device: torch.device,
    mode: str,
    mlflow_experiment: str = "neuroseg-H1-finetune",
    base_tags: Optional[dict] = None,
) -> dict:
    """
    Fine-tune or train a supervised baseline on labeled data.

    mode='finetune'            — loads pretrained JEPA weights first.
    mode='supervised_baseline' — random init, same architecture.

    Supports both Neurofinder format (auto-detected) and the custom
    video.tif + mask.tif layout.
    """
    labeled_path = Path(labeled_data_dir) if labeled_data_dir else Path("")
    if not labeled_path.exists():
        print(f"Labeled data not found at {labeled_path} — skipping {mode} f={fraction}")
        return {}

    dataset = _make_labeled_dataset(labeled_data_dir, cfg, fraction)
    if len(dataset) == 0:
        print(f"No labeled clips — skipping {mode} f={fraction}")
        return {}

    n_val = max(1, int(len(dataset) * cfg.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    jepa = build_jepa(cfg.arch_dict(), device)
    if mode == "finetune" and pretrained_checkpoint is not None:
        state_dict = torch.load(str(pretrained_checkpoint), map_location=device, weights_only=True)
        jepa.load_state_dict(state_dict)

    seg_head = build_seg_head(cfg.dstc, cfg.seg_head_hidden).to(device)
    encoder_lr = cfg.lr / 10 if mode == "finetune" else cfg.lr
    optimizer = Adam([
        {"params": jepa.encoder.parameters(), "lr": encoder_lr},
        {"params": seg_head.parameters(), "lr": cfg.lr},
    ])
    criterion = nn.BCELoss()

    tags = {"hypothesis": "H1", "mode": mode, "labeled_fraction": str(fraction)}
    if base_tags:
        tags.update(base_tags)
    model_name = tags.get("model_name", f"jepa_h1_{mode}_f{int(fraction * 100)}")

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(tags=tags) as run:
        mlflow.log_params({"labeled_fraction": fraction, "mode": mode,
                           "finetune_epochs": cfg.finetune_epochs})

        jepa.train()
        seg_head.train()
        val_dice_score, val_miou_score = 0.0, 0.0

        for epoch in range(cfg.finetune_epochs):
            epoch_loss, n_batches = 0.0, 0

            for batch in tqdm(train_loader, desc=f"[{mode}] f={fraction} ep={epoch}"):
                x = batch["video"].to(device)
                mask_gt = batch["mask"].to(device)

                enc_state = jepa.encoder(x)
                enc_mean = enc_state.mean(dim=2)
                pred = seg_head(enc_mean)
                pred = nn.functional.interpolate(
                    pred, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = criterion(pred, mask_gt[:, 0:1].float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            val_dice_score, val_miou_score = _validate_finetune(val_loader, jepa, seg_head, device)
            mlflow.log_metrics(
                {"train/loss": epoch_loss / max(n_batches, 1),
                 "val/dice": val_dice_score, "val/miou": val_miou_score},
                step=epoch,
            )

        run_id = run.info.run_id
        checkpoint_path = save_compound_checkpoint(
            models={"jepa": jepa, "seg_head": seg_head},
            arch=cfg.arch_dict(),
            model_name=model_name,
            run_id=run_id,
            output_dir=output_dir,
            metadata={**tags, "dice": val_dice_score, "miou": val_miou_score},
        )
        mlflow.log_artifact(str(checkpoint_path))

    return {"dice": val_dice_score, "miou": val_miou_score}


@torch.inference_mode()
def _validate_finetune(val_loader, jepa: JEPA, seg_head: nn.Module, device: torch.device) -> tuple[float, float]:
    jepa.eval()
    seg_head.eval()
    dice_scores, miou_scores = [], []

    for batch in val_loader:
        x = batch["video"].to(device)
        mask_gt = batch["mask"].to(device)

        enc_state = jepa.encoder(x)
        enc_mean = enc_state.mean(dim=2)
        pred = seg_head(enc_mean)
        pred = nn.functional.interpolate(
            pred, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False
        )

        pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        gt_np = (mask_gt[:, 0].squeeze().cpu().numpy() > 0).astype(np.uint8)

        dice_scores.append(dice(pred_np, gt_np))
        miou_scores.append(miou(pred_np, gt_np, num_classes=2))

    jepa.train()
    seg_head.train()
    if not dice_scores:
        return 0.0, 0.0
    return float(np.mean(dice_scores)), float(np.mean(miou_scores))


def run_h1(state: State):
    cfg = build_config(state)
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(state["output_dir"])
    data_dir = state["data_dir"]
    labeled_data_dir = state.get("config", {}).get("labeled_data_dir", "")
    fractions = state.get("config", {}).get("labeled_fractions", cfg.labeled_fractions)

    print(f"[H1] device={device} | data={data_dir} | output={output_dir}")

    pretrained_path = pretrain(
        data_dir, cfg, output_dir, device,
        mlflow_experiment="neuroseg-H1-pretrain",
        base_tags={"model_name": "jepa_pretrained_h1"},
    )

    for fraction in fractions:
        finetune(
            pretrained_path, labeled_data_dir, fraction, cfg, output_dir, device,
            mode="finetune",
            mlflow_experiment="neuroseg-H1-finetune",
            base_tags={"model_name": f"jepa_h1_finetune_f{int(fraction * 100)}"},
        )
        finetune(
            None, labeled_data_dir, fraction, cfg, output_dir, device,
            mode="supervised_baseline",
            mlflow_experiment="neuroseg-H1-finetune",
            base_tags={"model_name": f"jepa_h1_supervised_f{int(fraction * 100)}"},
        )

    print("[H1] Done.")
