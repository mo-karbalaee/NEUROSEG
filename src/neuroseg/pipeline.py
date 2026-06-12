from langgraph.graph import StateGraph, START, END
from neuroseg.models.state import State
from neuroseg.models.node import Node

def add_nodes(workflow: StateGraph):
    workflow.add_node(Node.LOADER)
    workflow.add_node(Node.PRE_PROCESSOR)
    workflow.add_node(Node.SEGMENTER)
    workflow.add_node(Node.ACTIVITY_TRACE_DETECTOR)
    workflow.add_node(Node.VISUALIZER)
    
def add_edges(workflow: StateGraph):
    pass

def visualize_pipeline():
    pass

def build_app():
    workflow = StateGraph(State)
    
    add_nodes(workflow)
    add_edges(workflow)

    app = workflow.compile()
