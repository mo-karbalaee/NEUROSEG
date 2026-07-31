from enum import StrEnum

class Node(StrEnum):
    """Named pipeline stages: the inference nodes plus the training node."""
    LOADER = "loader"
    PRE_PROCESSOR = "pre-processor"
    SEGMENTER = "segmenter"
    ACTIVITY_TRACE_CALCULATOR = "activity-trace-calculator"
    VISUALIZER = "visualizer"
    TRAINING = "training"
    
    