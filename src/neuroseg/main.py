from neuroseg.calcium_recording import CalciumRecording
from neuroseg.segmenter import Segmenter

recording: CalciumRecording = CalciumRecording(path="../../data/6s.tif")

masks, flows = Segmenter.generate_mask(recording)

recording.visualize(masks, flows)