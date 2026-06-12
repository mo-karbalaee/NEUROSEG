from typing import TypedDict, Optional
import numpy as np
from neuroseg.models.mode import Mode


class State(TypedDict):
    mode: Mode
    file_paths: list
    current_file_index: int
    file_name: Optional[str]
    data: Optional[np.ndarray]
    masks: Optional[list]
    flows: Optional[list]
    traces: Optional[np.ndarray]
