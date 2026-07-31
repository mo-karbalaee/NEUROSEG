from typing import TypedDict, Optional
import numpy as np
from neuroseg.models.mode import Mode
from neuroseg.models.hypothesis import Hypothesis


class State(TypedDict):
    """Shared LangGraph state schema carried through every pipeline node."""
    mode: Mode
    hypothesis: Optional[Hypothesis]
    data_dir: str
    output_dir: str
    checkpoint_path: Optional[str]
    config: dict
    file_paths: list
    current_file_index: int
    file_name: Optional[str]
    data: Optional[np.ndarray]
    masks: Optional[list]
    flows: Optional[list]
    traces: Optional[np.ndarray]
