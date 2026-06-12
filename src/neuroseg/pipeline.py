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


def _training_placeholder(_state: State):
    print("Training not implemented yet")


def _which_mode(state: State):
    return state["mode"]


def _files_remaining(state: State) -> str:
    if state["current_file_index"] < len(state["file_paths"]):
        return "continue"
    return "done"


def build_app():
    workflow = StateGraph(State)

    workflow.add_node(Node.LOADER, loader_node)
    workflow.add_node(Node.PRE_PROCESSOR, pre_processor_node)
    workflow.add_node(Node.SEGMENTER, segmenter_node)
    workflow.add_node(Node.ACTIVITY_TRACE_CALCULATOR, activity_trace_calculator_node)
    workflow.add_node(Node.VISUALIZER, visualizer_node)
    workflow.add_node(Node.TRAINING, _training_placeholder)

    workflow.add_conditional_edges(
        START, _which_mode, {Mode.INFERENCE: Node.LOADER, Mode.TRAINING: Node.TRAINING}
    )
    workflow.add_edge(Node.TRAINING, END)
    workflow.add_edge(Node.LOADER, Node.PRE_PROCESSOR)
    workflow.add_edge(Node.PRE_PROCESSOR, Node.SEGMENTER)
    workflow.add_edge(Node.SEGMENTER, Node.ACTIVITY_TRACE_CALCULATOR)
    workflow.add_edge(Node.ACTIVITY_TRACE_CALCULATOR, Node.VISUALIZER)
    workflow.add_conditional_edges(
        Node.VISUALIZER, _files_remaining, {"continue": Node.LOADER, "done": END}
    )

    return workflow.compile()


def run(data_dir: str | Path, mode: Mode = Mode.INFERENCE):
    data_dir = Path(data_dir)
    file_paths = [str(p) for p in sorted(data_dir.iterdir()) if p.is_file()]
    print(f"Found {len(file_paths)} file(s): {file_paths}")

    app = build_app()
    return app.invoke({
        "mode": mode,
        "file_paths": file_paths,
        "current_file_index": 0,
        "file_name": None,
        "data": None,
        "masks": None,
        "flows": None,
        "traces": None,
    })


def visualize_pipeline(output_path: str | Path = "docs/pipeline.png"):
    app = build_app()
    app.get_graph().draw_png(str(output_path))
