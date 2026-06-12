from enum import StrEnum

class Node(StrEnum):
    # inference nodes
    LOADER = "loader"
    PRE_PROCESSOR = "pre-processor"
    SEGMENTER = "segmenter"
    ACTIVITY_TRACE_CALCULATOR = "activity-trace-calculator"
    VISUALIZER = "visualizer"
    
    # training nodes
    TRAINING = "training"
    
    