import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from neuroseg.checkpoint import save_checkpoint, save_compound_checkpoint, save_latest_checkpoint
from neuroseg.logger import RunLogger
from neuroseg.metrics import dice, miou
from neuroseg.models.state import State
from neuroseg.trainers.dataset import (
    AugmentedClips,
    LabeledTIFFDataset,
    NeurofinderDataset,
    VideoFolderDataset,
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

LABELED_FRACTIONS = [0.1, 0.5, 0.75, 1.0]


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
    std_margin: float = 1.0
    ema_momentum: float = 0.996
    context_length: int = 2
    decoder_hidden_dim: int = 16
    seg_threshold: float = 0.5
    f0_percentile: int = 10
    dff_epsilon: float = 1e-6
    lr: float = 1e-3
    recon_coeff: float = 1.0
    finetune_encoder_lr_scale: float = 0.1
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    checkpoint_every: int = 0
    pretrain_augment: bool = True
    pretrain_patience: int = 0
    steps: int = 4
    seed: int = 1
    labeled_fractions: list[float] = field(default_factory=lambda: list(LABELED_FRACTIONS))
    val_split: float = 0.1
    test_split: float = 0.2
    pretrain_clip_stride: Optional[int] = None
    pretrain_max_file_gb: Optional[float] = None

    def arch_dict(self) -> dict:
        """Return architecture hyperparameters as a dict suitable for saving in checkpoints."""
        return {
            "dobs": self.dobs,
            "henc": self.henc,
            "hpre": self.hpre,
            "dstc": self.dstc,
            "seg_head_hidden": self.seg_head_hidden,
            "img_size": self.img_size,
            "context_length": self.context_length,
            "std_coeff": self.std_coeff,
            "cov_coeff": self.cov_coeff,
            "std_margin": self.std_margin,
            "ema_momentum": self.ema_momentum,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "seg_threshold": self.seg_threshold,
        }


def setup_seed(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(state: State) -> H1Config:
    """Build an H1Config by overlaying any matching keys from state['config']."""
    cfg = H1Config()
    for k, v in state.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _make_unlabeled_dataset(
    data_dir: str, cfg: H1Config, exclude_dirs: Optional[list] = None
) -> Dataset:
    """Create an unlabeled dataset from either Neurofinder or plain TIFF files."""
    if is_neurofinder_dir(data_dir) or find_neurofinder_dirs(data_dir):
        return NeurofinderDataset(
            data_dir, cfg.seq_len, cfg.img_size,
            labeled=False, clip_stride=cfg.pretrain_clip_stride,
            exclude_dirs=exclude_dirs,
        )
    return VideoFolderDataset(
        data_dir, cfg.seq_len, cfg.img_size, clip_stride=cfg.pretrain_clip_stride,
        max_file_gb=cfg.pretrain_max_file_gb,
    )


def _make_labeled_dataset(
    data_dir: str,
    cfg: H1Config,
    fraction: float,
    binarize: bool = True,
) -> Dataset:
    """Create a labeled dataset for fine-tuning, sampling the given labeled fraction."""
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
    model_name: str = "jepa_pretrained_h1",
    log_path: Optional[Path] = None,
    hypothesis: str = "H1",
    exclude_dirs: Optional[list] = None,
) -> Optional[Path]:
    """Self-supervised JEPA pretraining on unlabeled data; returns the checkpoint path.

    exclude_dirs holds recordings out of pretraining (e.g. the held-out test
    recording) so the self-supervised step never sees the downstream test data.
    """
    dataset = _make_unlabeled_dataset(data_dir, cfg, exclude_dirs=exclude_dirs)
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
    train_source = AugmentedClips(train_set) if cfg.pretrain_augment else train_set
    train_loader = DataLoader(
        train_source, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    print(
        f"[pretrain] clips={len(dataset)} train={len(train_set)} val={len(val_set)} "
        f"| augment={cfg.pretrain_augment} | dstc={cfg.dstc} | patience={cfg.pretrain_patience}"
    )

    jepa = build_jepa(cfg.arch_dict(), device)
    decoder = ImageDecoder(cfg.dstc, cfg.dobs, hidden_dim=cfg.decoder_hidden_dim)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss(), train_encoder=True).to(device)

    optimizer = Adam([
        {"params": jepa.parameters(), "lr": cfg.lr},
        {"params": pixel_decoder.head.parameters(), "lr": cfg.lr / 10},
    ])

    logger = RunLogger(
        log_path or output_dir / "logs" / "runs.csv",
        hypothesis=hypothesis, mode="pretrain", model_name=model_name,
    )

    jepa.train()
    pixel_decoder.train()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

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
            (jepa_loss + cfg.recon_coeff * recon_loss).backward()
            optimizer.step()
            jepa.update_target()

            epoch_jepa += jepa_loss.item()
            epoch_recon += recon_loss.item()
            n_batches += 1
            pbar.set_postfix(jepa=f"{jepa_loss.item():.4f}", recon=f"{recon_loss.item():.4f}")

        val_metrics = _validate_pretrain(val_loader, jepa, pixel_decoder, cfg, device)
        logger.log(
            epoch=epoch,
            train_loss=epoch_jepa / n_batches,
            train_recon_loss=epoch_recon / n_batches,
            **val_metrics,
        )

        val_jepa_loss = val_metrics["val_jepa_loss"]
        if val_jepa_loss < best_val:
            best_val = val_jepa_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in jepa.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
            save_latest_checkpoint(
                jepa, model_name, logger.run_id, output_dir,
                arch=cfg.arch_dict(),
                metadata={"hypothesis": hypothesis, "mode": "pretrain", "epoch": epoch},
            )

        if cfg.pretrain_patience and epochs_no_improve >= cfg.pretrain_patience:
            print(
                f"[pretrain] early stop at epoch {epoch} "
                f"(no val improvement for {cfg.pretrain_patience} epochs)"
            )
            break

    if best_state is not None:
        jepa.load_state_dict(best_state)
        print(f"[pretrain] restored best-val encoder from epoch {best_epoch} (val_jepa_loss={best_val:.4f})")

    checkpoint_path = save_checkpoint(
        jepa, model_name=model_name, run_id=logger.run_id,
        output_dir=output_dir,
        metadata={"hypothesis": hypothesis, "mode": "pretrain",
                  "best_epoch": best_epoch, "best_val_jepa_loss": best_val},
        arch=cfg.arch_dict(),
    )
    print(f"Pretrained checkpoint (best-val): {checkpoint_path}")

    from neuroseg.plots import plot_pretrain_curves
    plot_pretrain_curves(
        log_path or output_dir / "logs" / "runs.csv",
        logger.run_id, model_name, output_dir / "figures",
    )

    return checkpoint_path


@torch.inference_mode()
def _validate_pretrain(val_loader, jepa: JEPA, pixel_decoder, cfg: H1Config, device: torch.device) -> dict:
    """Evaluate JEPA and reconstruction losses on the validation set."""
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
        return {"val_jepa_loss": 0.0, "val_recon_loss": 0.0}
    return {"val_jepa_loss": float(np.mean(val_jepa)), "val_recon_loss": float(np.mean(val_recon))}


def finetune(
    pretrained_checkpoint: Optional[Path],
    labeled_data_dir: str,
    fraction: float,
    cfg: H1Config,
    output_dir: Path,
    device: torch.device,
    mode: str,
    model_name: Optional[str] = None,
    log_path: Optional[Path] = None,
    hypothesis: str = "H1",
) -> dict:
    """
    Fine-tune or train a supervised baseline on labeled data.

    mode='finetune'            — loads pretrained JEPA weights first.
    mode='supervised_baseline' — random init, same architecture.
    """
    labeled_path = Path(labeled_data_dir) if labeled_data_dir else Path("")
    if not labeled_path.exists():
        print(f"Labeled data not found at {labeled_path} — skipping {mode} f={fraction}")
        return {}

    dataset = _make_labeled_dataset(labeled_data_dir, cfg, fraction=1.0)
    if len(dataset) < 3:
        print(f"Too few labeled clips ({len(dataset)}) — skipping {mode} f={fraction}")
        return {}

    n = len(dataset)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    n_test = max(1, int(n * cfg.test_split))
    test_idx = perm[:n_test]
    pool = perm[n_test:]
    n_val = max(1, int(len(pool) * cfg.val_split))
    val_idx = pool[:n_val]
    train_pool = pool[n_val:]
    k = max(1, int(fraction * len(train_pool)))
    train_idx = train_pool[:k]

    test_set = Subset(dataset, test_idx)
    val_set = Subset(dataset, val_idx)
    train_set = Subset(dataset, train_idx)
    print(
        f"[{mode}] f={fraction} | clips={n} test={len(test_set)} (fixed) "
        f"val={len(val_set)} train={len(train_set)}/{len(train_pool)}"
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    jepa = build_jepa(cfg.arch_dict(), device)
    if mode == "finetune" and pretrained_checkpoint is not None:
        state_dict = torch.load(str(pretrained_checkpoint), map_location=device, weights_only=True)
        jepa.load_state_dict(state_dict, strict=False)

    seg_head = build_seg_head(cfg.dstc, cfg.seg_head_hidden).to(device)
    encoder_lr = cfg.lr * cfg.finetune_encoder_lr_scale if mode == "finetune" else cfg.lr
    optimizer = Adam([
        {"params": jepa.encoder.parameters(), "lr": encoder_lr},
        {"params": seg_head.parameters(), "lr": cfg.lr},
    ])
    criterion = nn.BCELoss()

    if model_name is None:
        model_name = f"jepa_h1_{mode}_f{int(fraction * 100)}"

    logger = RunLogger(
        log_path or output_dir / "logs" / "runs.csv",
        hypothesis=hypothesis, mode=mode, model_name=model_name,
        labeled_fraction=fraction,
    )

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
        logger.log(
            epoch=epoch,
            train_loss=epoch_loss / max(n_batches, 1),
            val_dice=val_dice_score,
            val_miou=val_miou_score,
        )

    test_dice_score, test_miou_score = _validate_finetune(test_loader, jepa, seg_head, device)

    actual_log_path = log_path or output_dir / "logs" / "runs.csv"
    checkpoint_path = save_compound_checkpoint(
        models={"jepa": jepa, "seg_head": seg_head},
        arch=cfg.arch_dict(),
        model_name=model_name,
        run_id=logger.run_id,
        output_dir=output_dir,
        metadata={"hypothesis": hypothesis, "mode": mode, "labeled_fraction": fraction,
                  "dice": test_dice_score, "miou": test_miou_score},
    )
    logger.log(test_dice=test_dice_score, test_miou=test_miou_score,
               checkpoint=str(checkpoint_path))
    print(f"[{mode}] f={fraction} | test_dice={test_dice_score:.4f}  test_miou={test_miou_score:.4f}")

    from neuroseg.plots import plot_finetune_curves
    plot_finetune_curves(actual_log_path, logger.run_id, model_name, output_dir / "figures")

    return {"dice": test_dice_score, "miou": test_miou_score, "checkpoint": str(checkpoint_path)}


@torch.inference_mode()
def _validate_finetune(val_loader, jepa: JEPA, seg_head: nn.Module, device: torch.device) -> tuple[float, float]:
    """Evaluate mean Dice and mIoU of the segmentation head on the validation set."""
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
    """Run the full H1 experiment: pretrain, then fine-tune and evaluate at each labeled fraction."""
    cfg = build_config(state)
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(state["output_dir"])
    data_dir = state["data_dir"]
    labeled_data_dir = state.get("config", {}).get("labeled_data_dir", "")
    fractions = state.get("config", {}).get("labeled_fractions", cfg.labeled_fractions)
    log_path = output_dir / "logs" / "runs.csv"

    exclude = [labeled_data_dir] if labeled_data_dir else None
    print(
        f"[H1] device={device} | data={data_dir} | output={output_dir}\n"
        f"[H1] pretrain excludes the test recording (unseen by SSL): {labeled_data_dir or '(none)'}\n"
        f"[H1] finetune/baseline + fixed 80/20 test split operate only on: {labeled_data_dir or '(none)'}"
    )

    if exclude:
        pool_err = (
            f"After excluding the test recording ({labeled_data_dir}), no clips remain for "
            f"self-supervised pretraining under --data={data_dir}. H1 needs at least one "
            f"recording OTHER than the test recording in --data. Point --data at a folder "
            f"containing the test recording PLUS other Neurofinder recordings."
        )
        try:
            remaining = _make_unlabeled_dataset(data_dir, cfg, exclude_dirs=exclude)
        except ValueError:
            raise ValueError(pool_err)
        if len(remaining) == 0:
            raise ValueError(pool_err)

    pretrained_path = pretrain(
        data_dir, cfg, output_dir, device,
        model_name="jepa_pretrained_h1",
        log_path=log_path,
        exclude_dirs=exclude,
    )
    if pretrained_path is None:
        raise ValueError(
            "Pretraining produced no checkpoint (empty pretrain pool). The 'finetune' arm "
            "would silently equal the supervised baseline. Check --data contains recordings "
            "beyond the excluded test recording."
        )

    for fraction in fractions:
        finetune(
            pretrained_path, labeled_data_dir, fraction, cfg, output_dir, device,
            mode="finetune",
            model_name=f"jepa_h1_finetune_f{int(fraction * 100)}",
            log_path=log_path,
        )
        finetune(
            None, labeled_data_dir, fraction, cfg, output_dir, device,
            mode="supervised_baseline",
            model_name=f"jepa_h1_supervised_f{int(fraction * 100)}",
            log_path=log_path,
        )

    print("[H1] Done.")

    from neuroseg.plots import plot_h1_dice
    plot_h1_dice(log_path, output_dir / "figures")
