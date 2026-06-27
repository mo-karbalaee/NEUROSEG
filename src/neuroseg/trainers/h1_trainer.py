import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from neuroseg.checkpoint import save_checkpoint
from neuroseg.metrics import dice, miou
from neuroseg.models.state import State
from neuroseg.trainers.dataset import LabeledTIFFDataset, TIFFVideoDataset
from neuroseg.trainers.jepa import (
    JEPA,
    ImageDecoder,
    JEPAProbe,
    Projector,
    ResNet5,
    ResUNet,
    SquareLossSeq,
    StateOnlyPredictor,
    VCLoss,
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
    std_coeff: float = 10.0
    cov_coeff: float = 100.0
    lr: float = 1e-3
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    steps: int = 4
    seed: int = 1
    labeled_fractions: list[float] = field(default_factory=lambda: list(LABELED_FRACTIONS))
    val_split: float = 0.1


def _setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_config(state: State) -> H1Config:
    cfg = H1Config()
    for k, v in state.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _build_jepa(cfg: H1Config, device: torch.device) -> tuple[JEPA, JEPAProbe]:
    encoder = ResNet5(cfg.dobs, cfg.henc, cfg.dstc)
    predictor_model = ResUNet(2 * cfg.dstc, cfg.hpre, cfg.dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{cfg.dstc}-{cfg.dstc * 4}-{cfg.dstc * 4}")
    regularizer = VCLoss(cfg.std_coeff, cfg.cov_coeff, proj=projector)
    ploss_fn = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss_fn).to(device)

    decoder = ImageDecoder(cfg.dstc, cfg.dobs, hidden_dim=16)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)

    return jepa, pixel_decoder


def _pretrain(
    file_paths: list[str],
    cfg: H1Config,
    output_dir: Path,
    device: torch.device,
) -> Optional[Path]:
    dataset = TIFFVideoDataset(file_paths, seq_len=cfg.seq_len, img_size=cfg.img_size)
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

    jepa, pixel_decoder = _build_jepa(cfg, device)
    optimizer = Adam([
        {"params": jepa.parameters(), "lr": cfg.lr},
        {"params": pixel_decoder.head.parameters(), "lr": cfg.lr / 10},
    ])

    mlflow.set_experiment("neuroseg-H1-pretrain")
    with mlflow.start_run(tags={"hypothesis": "H1", "mode": "pretrain"}) as run:
        mlflow.log_params({
            "seq_len": cfg.seq_len,
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "henc": cfg.henc,
            "hpre": cfg.hpre,
            "dstc": cfg.dstc,
            "std_coeff": cfg.std_coeff,
            "cov_coeff": cfg.cov_coeff,
            "lr": cfg.lr,
            "pretrain_epochs": cfg.pretrain_epochs,
            "steps": cfg.steps,
        })

        jepa.train()
        pixel_decoder.train()

        for epoch in range(cfg.pretrain_epochs):
            epoch_jepa_loss = 0.0
            epoch_recon_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"[H1 pretrain] Epoch {epoch}")
            for batch in pbar:
                x = batch["video"].to(device)
                optimizer.zero_grad()

                _, (jepa_loss, regl, _, regl_dict, pl) = jepa.unroll(
                    x, None, nsteps=cfg.steps, unroll_mode="parallel", compute_loss=True
                )
                recon_loss = pixel_decoder(x, x)
                total_loss = jepa_loss + recon_loss

                total_loss.backward()
                optimizer.step()

                epoch_jepa_loss += jepa_loss.item()
                epoch_recon_loss += recon_loss.item()
                n_batches += 1
                pbar.set_postfix(
                    jepa=f"{jepa_loss.item():.4f}", recon=f"{recon_loss.item():.4f}"
                )

            train_metrics = {
                "train/jepa_loss": epoch_jepa_loss / n_batches,
                "train/recon_loss": epoch_recon_loss / n_batches,
            }

            val_metrics = _validate_pretrain(val_loader, jepa, pixel_decoder, cfg, device)
            mlflow.log_metrics({**train_metrics, **val_metrics}, step=epoch)

        run_id = run.info.run_id
        checkpoint_path = save_checkpoint(
            jepa,
            model_name="jepa_pretrained_h1",
            run_id=run_id,
            output_dir=output_dir,
            metadata={"hypothesis": "H1", "mode": "pretrain"},
        )
        mlflow.log_artifact(str(checkpoint_path))
        print(f"Pretrained checkpoint saved: {checkpoint_path}")

    return checkpoint_path


@torch.inference_mode()
def _validate_pretrain(val_loader, jepa, pixel_decoder, cfg: H1Config, device: torch.device) -> dict:
    jepa.eval()
    pixel_decoder.eval()
    val_recon = []
    val_jepa = []

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
    return {
        "val/jepa_loss": float(np.mean(val_jepa)),
        "val/recon_loss": float(np.mean(val_recon)),
    }


def _finetune(
    pretrained_checkpoint: Optional[Path],
    labeled_data_dir: str,
    fraction: float,
    cfg: H1Config,
    output_dir: Path,
    device: torch.device,
    mode: str,
) -> dict:
    """
    Fine-tune (or train from scratch for supervised baseline) on labeled data.

    `mode` is either 'finetune' (loads pretrained weights) or 'supervised_baseline'
    (random init). Returns final dice and miou across the validation split.

    Labeled data is expected at `labeled_data_dir` in the LabeledTIFFDataset format:
        labeled_data_dir/
            sample_001/
                video.tif
                mask.tif
            ...
    """
    labeled_data_path = Path(labeled_data_dir)
    if not labeled_data_path.exists():
        print(f"Labeled data not found at {labeled_data_path} — skipping {mode} fraction={fraction}")
        return {}

    dataset = LabeledTIFFDataset(
        str(labeled_data_path),
        seq_len=cfg.seq_len,
        img_size=cfg.img_size,
        labeled_fraction=fraction,
        seed=cfg.seed,
    )
    if len(dataset) == 0:
        print(f"No labeled samples found — skipping {mode} fraction={fraction}")
        return {}

    n_val = max(1, int(len(dataset) * cfg.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    jepa, _ = _build_jepa(cfg, device)

    if mode == "finetune" and pretrained_checkpoint is not None:
        state_dict = torch.load(str(pretrained_checkpoint), map_location=device)
        jepa.load_state_dict(state_dict)

    seg_head = nn.Sequential(
        nn.Conv2d(cfg.dstc, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 1),
        nn.Sigmoid(),
    ).to(device)

    optimizer = Adam([
        {"params": jepa.encoder.parameters(), "lr": cfg.lr / 10},
        {"params": seg_head.parameters(), "lr": cfg.lr},
    ])
    criterion = nn.BCELoss()

    model_name = f"jepa_h1_{mode}_f{int(fraction * 100)}"
    mlflow.set_experiment("neuroseg-H1-finetune")
    with mlflow.start_run(
        tags={"hypothesis": "H1", "mode": mode, "labeled_fraction": str(fraction)}
    ) as run:
        mlflow.log_params({
            "labeled_fraction": fraction,
            "mode": mode,
            "finetune_epochs": cfg.finetune_epochs,
        })

        jepa.train()
        seg_head.train()

        for epoch in range(cfg.finetune_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"[H1 {mode}] f={fraction} epoch {epoch}"):
                x = batch["video"].to(device)
                mask = batch["mask"].to(device)

                with torch.no_grad():
                    enc_state = jepa.encoder(x)

                enc_mean = enc_state.mean(dim=2)
                pred_mask = seg_head(enc_mean)
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask, size=mask.shape[-2:], mode="bilinear", align_corners=False
                )
                target = mask[:, 0:1].float()
                loss = criterion(pred_mask, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            val_dice, val_miou = _validate_finetune(val_loader, jepa, seg_head, device)
            mlflow.log_metrics(
                {
                    "train/loss": epoch_loss / max(n_batches, 1),
                    "val/dice": val_dice,
                    "val/miou": val_miou,
                },
                step=epoch,
            )

        run_id = run.info.run_id
        checkpoint_path = save_checkpoint(
            jepa,
            model_name=model_name,
            run_id=run_id,
            output_dir=output_dir,
            metadata={
                "hypothesis": "H1",
                "mode": mode,
                "labeled_fraction": fraction,
                "dice": val_dice,
                "miou": val_miou,
            },
        )
        mlflow.log_artifact(str(checkpoint_path))

    return {"dice": val_dice, "miou": val_miou}


@torch.inference_mode()
def _validate_finetune(val_loader, jepa, seg_head, device: torch.device) -> tuple[float, float]:
    jepa.eval()
    seg_head.eval()
    dice_scores = []
    miou_scores = []

    for batch in val_loader:
        x = batch["video"].to(device)
        mask = batch["mask"].to(device)

        enc_state = jepa.encoder(x)
        enc_mean = enc_state.mean(dim=2)
        pred_mask = seg_head(enc_mean)
        pred_mask = torch.nn.functional.interpolate(
            pred_mask, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )

        pred_np = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        gt_np = (mask[:, 0].squeeze().cpu().numpy() > 0).astype(np.uint8)

        dice_scores.append(dice(pred_np, gt_np))
        miou_scores.append(miou(pred_np, gt_np, num_classes=2))

    jepa.train()
    seg_head.train()
    return float(np.mean(dice_scores)), float(np.mean(miou_scores))


def run_h1(state: State):
    cfg = _build_config(state)
    _setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(state["output_dir"])
    file_paths = state["file_paths"]
    labeled_data_dir = state.get("config", {}).get("labeled_data_dir", "")

    print(f"[H1] Device: {device} | Files: {len(file_paths)} | Output: {output_dir}")

    pretrained_path = _pretrain(file_paths, cfg, output_dir, device)

    fractions = state.get("config", {}).get(
        "labeled_fractions", cfg.labeled_fractions
    )
    for fraction in fractions:
        _finetune(
            pretrained_path, labeled_data_dir, fraction, cfg, output_dir, device, "finetune"
        )
        _finetune(
            None, labeled_data_dir, fraction, cfg, output_dir, device, "supervised_baseline"
        )

    print("[H1] Done.")
