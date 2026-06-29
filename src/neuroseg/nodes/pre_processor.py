import numpy as np
from neuroseg.models.state import State


def pre_processor_node(state: State) -> dict:
    """Normalize the loaded data to [0, 1] using per-stack min-max scaling."""
    data = state["data"].astype(np.float32)
    mn = float(data.min())
    mx = float(data.max())
    normalized = (data - mn) / (mx - mn + 1e-8)
    return {"data": normalized}
