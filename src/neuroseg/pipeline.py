from langgraph.graph import StateGraph, START, END
from neuroseg.models.state import State
from neuroseg.models.node import Node
from neuroseg.models.mode import Mode
from langchain_core.runnables.graph import MermaidDrawMethod


def add_nodes(workflow: StateGraph):
    workflow.add_node(Node.LOADER, placeholder)
    workflow.add_node(Node.PRE_PROCESSOR, placeholder)
    workflow.add_node(Node.SEGMENTER, placeholder)
    workflow.add_node(Node.ACTIVITY_TRACE_CALCULATOR, placeholder)
    workflow.add_node(Node.VISUALIZER, placeholder)
    workflow.add_node(Node.TRAINING, placeholder)


def placeholder(state: State):
    print("to be implemented")


def add_edges(workflow: StateGraph):
    workflow.add_conditional_edges(
        START, which_mode, {Mode.INFERENCE: Node.LOADER, Mode.TRAINING: Node.TRAINING}
    )

    workflow.add_edge(Node.TRAINING, END)

    workflow.add_edge(Node.LOADER, Node.PRE_PROCESSOR)
    workflow.add_edge(Node.PRE_PROCESSOR, Node.SEGMENTER)
    workflow.add_edge(Node.SEGMENTER, Node.ACTIVITY_TRACE_CALCULATOR)
    workflow.add_edge(Node.ACTIVITY_TRACE_CALCULATOR, Node.VISUALIZER)
    workflow.add_edge(Node.VISUALIZER, END)


def visualize_pipeline(app):
    png_bytes = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER
    )
    with open("docs/pipeline.png", "wb") as f:
        f.write(png_bytes)


def which_mode(state: State):
    return state["mode"]


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
app.invoke({"mode": parse_mode()})
