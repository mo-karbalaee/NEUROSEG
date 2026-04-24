from pathlib import Path
import os
from neuroseg.calcium_recording import CalciumRecording
from neuroseg.segmenter import Segmenter

data_dir = Path("../../data/")
file_paths = [p for p in data_dir.iterdir() if p.is_file()]

for file_path in file_paths:
    recording: CalciumRecording = CalciumRecording(path=str(file_path))
    print("Processing ", recording.file_name)

    if os.path.exists(os.path.join("../../output", f"masks+{recording.file_name}.npy")):
        print("Masks already processed")
        masks, flows = Segmenter.load_results(recording.file_name)
    else:
        print("Generating mask for ", recording.file_name)
        masks, flows = Segmenter.generate_mask(recording)
        Segmenter.save_results(masks, flows, recording.file_name)

    Segmenter.save_traces(Segmenter.extract_traces(masks, recording), recording.file_name)
    recording.visualize(masks, flows)

    print("Finished processing ", recording.file_name)
