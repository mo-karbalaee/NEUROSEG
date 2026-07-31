from enum import StrEnum


class Hypothesis(StrEnum):
    """The three experimental hypotheses selected via CLI flag (H1, H2, H3)."""
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
