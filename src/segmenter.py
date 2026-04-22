from cellpose import models

from models.calcium_recording import CalciumRecording


class Segmenter:
    def __init__(self):
        pass

    @staticmethod
    def generate_mask(recording: CalciumRecording):
        model = models.CellposeModel()

        masks, flows, styles = model.eval(recording.data, diameter=None)

        return masks, flows, styles

