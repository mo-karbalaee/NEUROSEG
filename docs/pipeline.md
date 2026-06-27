# Pipeline Architecture

## LangGraph StateGraph

NEUROSEG uses **LangGraph** (`langgraph`) to define both the inference and training pipelines as a single compiled `StateGraph`. This gives the pipeline a clear node-and-edge graph structure that can be visualised, reasoned about independently per node, and extended without modifying other nodes.

The graph is compiled once (`build_app()`) and invoked with an initial state dict. LangGraph handles routing between nodes and accumulation of state updates.

---

## The Pipeline State

All data flows through a single typed state dict (`src/neuroseg/models/state.py`):

```python
class State(TypedDict):
    mode: Mode                    # TRAINING or INFERENCE
    hypothesis: Optional[Hypothesis]  # H1, H2, or H3 (training only)
    data_dir: str                 # root data directory
    output_dir: str               # root output directory
    checkpoint_path: Optional[str]  # JEPA compound checkpoint (inference only)
    config: dict                  # merged YAML + CLI config
    file_paths: list              # all TIFF files found in data_dir
    current_file_index: int       # which file is being processed
    file_name: Optional[str]      # basename of the current file
    data: Optional[np.ndarray]    # loaded video (T × H × W)
    masks: Optional[list]         # segmentation masks (one per frame)
    flows: Optional[list]         # Cellpose flow fields (or None for JEPA)
    traces: Optional[np.ndarray]  # ΔF/F₀ traces (N neurons × T frames)
```

Every node returns a `dict` containing only the keys it modifies. LangGraph merges these updates into the running state.

---

## Graph Topology

```
                    ┌──────────┐
              START ┤          ├─── mode=TRAINING ───► TRAINING ──► END
                    │   mode   │
                    └──────────┘
                          │
                     mode=INFERENCE
                          ▼
                       LOADER
                          │
                    PRE_PROCESSOR
                          │
                      SEGMENTER
                          │
               ACTIVITY_TRACE_CALCULATOR
                          │
                       VISUALIZER
                          │
              ┌───────────┴────────────┐
         files_remaining          done
              │                        │
           LOADER                     END
```

### Conditional edges

**From START:** Routes on `state["mode"]`. `Mode.TRAINING` → `TRAINING` → `END` in one step. `Mode.INFERENCE` → the multi-node inference loop.

**From VISUALIZER:** Routes on `state["current_file_index"] < len(state["file_paths"])`. If files remain, loops back to `LOADER` to process the next file. When all files are processed, routes to `END`.

This loop structure means the inference pipeline processes every TIFF file in the input directory sequentially with a single `app.invoke()` call.

---

## Inference Pipeline Nodes

### LOADER (`nodes/loader.py`)

Reads the current file from disk:

- Accepts multi-frame TIFF (via `tifffile`) or AVI (via OpenCV).
- Handles edge cases: 2-D frames are treated as single-frame stacks; 4-D arrays are sliced to 3-D.
- Increments `current_file_index` so the next loop iteration picks the next file.

**State inputs:** `file_paths`, `current_file_index`  
**State outputs:** `data` (T × H × W ndarray), `file_name`, `current_file_index` (+1)

---

### PRE_PROCESSOR (`nodes/pre_processor.py`)

Currently a pass-through placeholder. Intended for any normalisation or augmentation that should be applied uniformly to all video data before segmentation. The separation into its own node makes it easy to add preprocessing steps (e.g., motion correction, background subtraction) without touching other nodes.

**State inputs:** `data`  
**State outputs:** `data` (unchanged)

---

### SEGMENTER (`nodes/segmenter.py`)

Applies one of two segmentation backends:

**JEPA segmenter** (when `checkpoint_path` is set):
1. Loads the compound checkpoint: reads `arch` dict, reconstructs `JEPA` and `seg_head` using factory functions.
2. For each frame `t`:
   - Adds batch and channel dimensions: `(1, 1, 1, H, W)`.
   - Encodes: `enc_state = jepa.encoder(x)` → `(1, dstc, 1, H, W)`.
   - Pools: `enc_mean = enc_state[:, :, 0]` → `(1, dstc, H, W)`.
   - Predicts: `pred = seg_head(enc_mean)` → `(1, 1, H, W)` in [0,1].
   - Upsamples to original resolution with bilinear interpolation.
   - Thresholds at 0.5 to get a binary mask.
   - Runs `scipy.ndimage.label` for connected-component labelling (each connected component gets a unique integer ID).
3. Returns a list of `(H, W)` integer label arrays, one per frame.

**Cellpose segmenter** (when no checkpoint):
- Delegates to `CellposeModel.eval()` with automatic diameter detection.
- Returns cell instance masks and flow fields.

**Caching:** Results are written to `output/cache/<model_key>/masks+<file>.npy` (and a corresponding `.pkl` for flows). On re-run the node checks for the cache and skips recomputation. The cache key is `"cellpose"` for the Cellpose backend or `"jepa_<checkpoint_stem>"` for JEPA.

**State inputs:** `data`, `checkpoint_path`, `output_dir`, `file_name`  
**State outputs:** `masks`, `flows`

---

### ACTIVITY_TRACE_CALCULATOR (`nodes/activity_trace_calculator.py`)

Extracts ΔF/F₀ fluorescence traces from the segmentation masks:

1. **Reference mask selection:** Picks the frame whose mask contains the most instances (`argmax(max(mask))`). This frame is used as the canonical neuron layout, avoiding flickering IDs across frames.
2. **Trace extraction:** For each neuron `n` and frame `t`, averages the raw pixel values inside neuron `n`'s mask region.
3. **ΔF/F₀ normalisation:**
   ```
   F0 = percentile(traces, 10, axis=time)
   ΔF/F₀ = (traces - F0) / (F0 + ε)
   ```
   The 10th-percentile baseline (`F0`) approximates the resting fluorescence. ΔF/F₀ represents the fractional change relative to baseline — the standard metric for calcium imaging signal strength.
4. Saves the raw trace matrix as `output/traces/traces+<file>.npy` (shape: N × T).

**State inputs:** `masks`, `data`, `output_dir`, `file_name`  
**State outputs:** `traces` (N × T ndarray)

---

### VISUALIZER (`nodes/visualizer.py`)

Saves two types of visualisations:

**Segmentation overlays** (`output/segmentation/<file>/frame_N.png`):
- Each frame is shown in greyscale.
- Each segmented neuron is overlaid with a semi-transparent colour (distinct hue per neuron ID, cycling through `tab20`).
- When Cellpose is used, the official `cellpose.plot.show_segmentation()` function renders flow fields too.

**Activity traces** (`output/traces/<file>/`):
- `traces_combined.png`: All N neuron traces overlaid on a single axes (ΔF/F₀ vs frame index).
- `neuron_N.png`: One trace per neuron in its own figure for clean inspection.

**State inputs:** `data`, `masks`, `flows`, `traces`, `output_dir`, `file_name`  
**State outputs:** `{}` (writes files, no state changes)

---

## Training Mode

When `mode=TRAINING`, the graph routes from `START` directly to the `TRAINING` node:

```python
def _training_node(state: State) -> dict:
    if hypothesis == Hypothesis.H1:
        run_h1(state)
    elif hypothesis == Hypothesis.H2:
        run_h2(state)
    elif hypothesis == Hypothesis.H3:
        run_h3(state)
    return {}
```

Each trainer function reads `state["data_dir"]`, `state["output_dir"]`, and `state["config"]`. It runs its own training loop independently and saves checkpoints + MLflow logs. The state is not modified (training is a side-effecting operation, not a data-passing one).

---

## Invocation

```python
from neuroseg.pipeline import run
from neuroseg.models.mode import Mode
from neuroseg.models.hypothesis import Hypothesis

run(
    data_dir="/path/to/tiffs",
    output_dir="/path/to/results",
    mode=Mode.TRAINING,
    hypothesis=Hypothesis.H1,
    config={"pretrain_epochs": 50, "labeled_data_dir": "/path/to/labeled"},
)
```

The `run()` function:
1. Lists all files in `data_dir` and places them in `file_paths`.
2. Builds and compiles the `StateGraph`.
3. Calls `app.invoke(initial_state)`, which runs the graph to completion.

---

## Visualising the Graph

```bash
uv run python -c "from neuroseg.pipeline import visualize_pipeline; visualize_pipeline()"
```

Saves `docs/pipeline.png` (requires `pygraphviz`; silently skips if not installed).
