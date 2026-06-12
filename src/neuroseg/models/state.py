from typing import TypedDict
from neuroseg.models.mode import Mode

class State(TypedDict):
    mode: Mode
    