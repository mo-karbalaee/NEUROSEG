from enum import Enum

class Node(Enum):
    LOADER = "loader"
    PRE_PROCESSOR = "pre-processor"
    SEGMENTER = "segmenter"
    ACTIVITY_TRACE_CALCULATOR = "activity-trace-calculator"
    VISUALIZER = "visualizer"