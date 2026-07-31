from enum import StrEnum


class Mode(StrEnum):
    """Top-level pipeline mode: training or inference."""
    TRAINING = "training"
    INFERENCE = "inference"
