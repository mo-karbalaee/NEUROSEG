from pathlib import Path
from langgraph.graph import StateGraph, START, END
from neuroseg.models.state import State
from neuroseg.models.node import Node
from neuroseg.models.mode import Mode
from neuroseg.nodes.loader import loader_node
from neuroseg.nodes.pre_processor import pre_processor_node
from neuroseg.nodes.segmenter import segmenter_node
from neuroseg.nodes.activity_trace_calculator import activity_trace_calculator_node
from neuroseg.nodes.visualizer import visualizer_node
from langchain_core.runnables.graph import MermaidDrawMethod


def add_nodes(workflow: StateGraph):
    workflow.add_node(Node.LOADER, loader_node)
    workflow.add_node(Node.PRE_PROCESSOR, pre_processor_node)
    workflow.add_node(Node.SEGMENTER, segmenter_node)
    workflow.add_node(Node.ACTIVITY_TRACE_CALCULATOR, activity_trace_calculator_node)
    workflow.add_node(Node.VISUALIZER, visualizer_node)
    workflow.add_node(Node.TRAINING, training_placeholder)


def training_placeholder(_state: State):
    print("Training not implemented yet")


def add_edges(workflow: StateGraph):
    workflow.add_conditional_edges(
        START, which_mode, {Mode.INFERENCE: Node.LOADER, Mode.TRAINING: Node.TRAINING}
    )

    workflow.add_edge(Node.TRAINING, END)

    workflow.add_edge(Node.LOADER, Node.PRE_PROCESSOR)
    workflow.add_edge(Node.PRE_PROCESSOR, Node.SEGMENTER)
    workflow.add_edge(Node.SEGMENTER, Node.ACTIVITY_TRACE_CALCULATOR)
    workflow.add_edge(Node.ACTIVITY_TRACE_CALCULATOR, Node.VISUALIZER)
    workflow.add_conditional_edges(
        Node.VISUALIZER, files_remaining, {"continue": Node.LOADER, "done": END}
    )


def which_mode(state: State):
    return state["mode"]


def files_remaining(state: State) -> str:
    if state["current_file_index"] < len(state["file_paths"]):
        return "continue"
    return "done"


def visualize_pipeline(app):
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("docs/pipeline.png", "wb") as f:
        f.write(png_bytes)


def build_app():
    workflow = StateGraph(State)
    add_nodes(workflow)
    add_edges(workflow)
    return workflow.compile()


import argparse


def parse_mode() -> Mode:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inference", action="store_true")
    group.add_argument("--training", action="store_true")
    args = parser.parse_args()
    return Mode.INFERENCE if args.inference else Mode.TRAINING


app = build_app()
# visualize_pipeline(app)

data_dir = Path("data/")
file_paths = [str(p) for p in sorted(data_dir.iterdir()) if p.is_file()]

app.invoke({
    "mode": parse_mode(),
    "file_paths": file_paths,
    "current_file_index": 0,
    "file_name": None,
    "data": None,
    "masks": None,
    "flows": None,
    "traces": None,
})
