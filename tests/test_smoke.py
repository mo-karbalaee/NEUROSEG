"""
Smoke tests — verify the package imports, core modules initialise, and
basic forward passes produce the expected output shapes.  No real data
or GPU required.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ── imports ──────────────────────────────────────────────────────────────────

def test_package_imports():
    from neuroseg.models.mode import Mode
    from neuroseg.models.hypothesis import Hypothesis
    from neuroseg.models.state import State
    from neuroseg.metrics import dice, miou
    from neuroseg.checkpoint import (
        save_compound_checkpoint,
        load_compound_checkpoint,
        list_checkpoints,
    )
    from neuroseg.trainers.jepa import build_jepa, build_seg_head
    from neuroseg.trainers.dataset import (
        is_neurofinder_dir,
        find_neurofinder_dirs,
        NeurofinderDataset,
    )
    from neuroseg.trainers.h1_trainer import H1Config, build_config


# ── JEPA model ───────────────────────────────────────────────────────────────

@pytest.fixture
def arch():
    return {"dobs": 1, "henc": 8, "hpre": 8, "dstc": 4, "seg_head_hidden": 4}


@pytest.fixture
def device():
    return torch.device("cpu")


def test_build_jepa(arch, device):
    from neuroseg.trainers.jepa import build_jepa
    jepa = build_jepa(arch, device)
    assert jepa is not None


def test_jepa_encoder_shape(arch, device):
    from neuroseg.trainers.jepa import build_jepa
    jepa = build_jepa(arch, device)
    x = torch.randn(2, 1, 5, 32, 32)  # (B, C, T, H, W)
    out = jepa.encoder(x)
    assert out.shape == (2, arch["dstc"], 5, 32, 32)


def test_jepa_unroll(arch, device):
    from neuroseg.trainers.jepa import build_jepa
    jepa = build_jepa(arch, device)
    x = torch.randn(2, 1, 6, 32, 32)
    _, losses = jepa.unroll(x, None, nsteps=2, unroll_mode="parallel", compute_loss=True)
    total_loss = losses[0]
    assert total_loss.item() > 0


def test_build_seg_head(arch):
    from neuroseg.trainers.jepa import build_seg_head
    head = build_seg_head(arch["dstc"], arch["seg_head_hidden"])
    x = torch.randn(2, arch["dstc"], 32, 32)
    out = head(x)
    assert out.shape == (2, 1, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0


# ── metrics ──────────────────────────────────────────────────────────────────

def test_dice_perfect():
    from neuroseg.metrics import dice
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    assert dice(mask, mask) == pytest.approx(1.0)


def test_dice_no_overlap():
    from neuroseg.metrics import dice
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt   = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    assert dice(pred, gt) == pytest.approx(0.0)


def test_miou_perfect():
    from neuroseg.metrics import miou
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert miou(mask, mask, num_classes=2) == pytest.approx(1.0)


def test_dice_all_zeros():
    from neuroseg.metrics import dice
    z = np.zeros((4, 4), dtype=np.uint8)
    assert dice(z, z) == pytest.approx(1.0)


# ── checkpoint round-trip ─────────────────────────────────────────────────────

def test_compound_checkpoint_roundtrip(arch, device):
    from neuroseg.trainers.jepa import build_jepa, build_seg_head
    from neuroseg.checkpoint import save_compound_checkpoint, load_compound_checkpoint

    jepa = build_jepa(arch, device)
    seg_head = build_seg_head(arch["dstc"], arch["seg_head_hidden"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_compound_checkpoint(
            models={"jepa": jepa, "seg_head": seg_head},
            arch=arch,
            model_name="test_model",
            run_id="abc123",
            output_dir=Path(tmpdir),
            metadata={"hypothesis": "H1", "dice": 0.85, "miou": 0.72},
        )
        payload = load_compound_checkpoint(path)

    assert payload["type"] == "neuroseg_jepa_v1"
    assert "jepa" in payload
    assert "seg_head" in payload
    assert payload["arch"] == arch


def test_list_checkpoints_filters_pretrain_only(arch, device):
    from neuroseg.trainers.jepa import build_jepa
    from neuroseg.checkpoint import save_checkpoint, save_compound_checkpoint, list_checkpoints
    from neuroseg.trainers.jepa import build_seg_head

    jepa = build_jepa(arch, device)
    seg_head = build_seg_head(arch["dstc"], arch["seg_head_hidden"])

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        save_checkpoint(jepa, "pretrain_only", "run1", out, metadata={"hypothesis": "H1"})
        save_compound_checkpoint(
            {"jepa": jepa, "seg_head": seg_head}, arch, "finetune_model", "run2", out,
            metadata={"hypothesis": "H1", "compound": True},
        )
        found = list_checkpoints(out)

    assert len(found) == 1
    assert found[0]["model_name"] == "finetune_model"


# ── dataset helpers ───────────────────────────────────────────────────────────

def test_is_neurofinder_dir_false(tmp_path):
    from neuroseg.trainers.dataset import is_neurofinder_dir
    assert not is_neurofinder_dir(tmp_path)


def test_is_neurofinder_dir_true(tmp_path):
    from neuroseg.trainers.dataset import is_neurofinder_dir
    (tmp_path / "images").mkdir()
    assert is_neurofinder_dir(tmp_path)


def test_find_neurofinder_dirs_single(tmp_path):
    from neuroseg.trainers.dataset import find_neurofinder_dirs
    (tmp_path / "images").mkdir()
    result = find_neurofinder_dirs(tmp_path)
    assert result == [tmp_path]


def test_find_neurofinder_dirs_multi(tmp_path):
    from neuroseg.trainers.dataset import find_neurofinder_dirs
    for name in ["neurofinder.00.00", "neurofinder.04.00"]:
        d = tmp_path / name
        d.mkdir()
        (d / "images").mkdir()
    result = find_neurofinder_dirs(tmp_path)
    assert len(result) == 2


# ── H1Config ─────────────────────────────────────────────────────────────────

def test_h1config_arch_dict():
    from neuroseg.trainers.h1_trainer import H1Config
    cfg = H1Config(dobs=1, henc=16, hpre=16, dstc=4)
    d = cfg.arch_dict()
    assert d["dstc"] == 4
    assert d["henc"] == 16


def test_build_config_from_state():
    from neuroseg.trainers.h1_trainer import build_config
    from neuroseg.models.mode import Mode
    state = {
        "mode": Mode.TRAINING,
        "hypothesis": None,
        "data_dir": "/tmp",
        "output_dir": "/tmp",
        "checkpoint_path": None,
        "config": {"seq_len": 7, "img_size": 64, "pretrain_epochs": 5},
        "file_paths": [],
        "current_file_index": 0,
        "file_name": None,
        "data": None,
        "masks": None,
        "flows": None,
        "traces": None,
    }
    cfg = build_config(state)
    assert cfg.seq_len == 7
    assert cfg.img_size == 64
    assert cfg.pretrain_epochs == 5
