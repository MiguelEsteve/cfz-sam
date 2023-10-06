import os

import onnx
import torch
from configs import configs
from fastsam import FastSAM, FastSAMPrompt

class ConversionFastSAM:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.model = self._get_model()
        self.prompt_process: FastSAMPrompt = object

    @property
    def checkpoint(self):
        return os.path.join(configs.PROJECT_PATH, 'FastSAM/weights/FastSAM.pt')

    def _get_model(self, checkpoint: str = None):
        checkpoint = self.checkpoint if checkpoint is None else checkpoint
        model = FastSAM(checkpoint)
        model.to(self.device)
        return model

    def export_to_ONNX(self):
        model = self._get_model()
        model.export(format='onnx')


    def load_from_ONNX(self):
        # load the ONNX model
        model = onnx.load('c:/repos/cfz-sam/FastSAM/weights/FastSAM.onnx')
        # check that the model is well formed
        onnx.checker.check_model(model)
        # print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))

    def inference_with_ONNX(self, input_data):
        import onnxruntime as ort

        ort_session = ort.InferenceSession('c:/repos/cfz-sam/FastSAM/weights/FastSAM.onnx',
                                           providers=['CPUExecutionProvider'])
        input_name  = ort_session.get_inputs()[0].name
        output_name  = ort_session.get_outputs()[0].name

        results = ort_session.run(
            None, {input_name: input_data}
        )
        return results
