import triton_python_backend_utils as pb_utils
from PIL import Image
import io
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TritonPythonModel:
    def initialize(self, args):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

    def execute(self, requests):
        responses = []
        for request in requests:
            # ✅ Extract bytes properly
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_image")
            input_data = input_tensor.as_numpy()[0]   # (1,) array of bytes
            img = Image.open(io.BytesIO(input_data)).convert("RGB")

            # Run OCR
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
            outputs = self.model.generate(pixel_values)
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # ✅ Encode output text back to BYTES
            out_tensor = pb_utils.Tensor("text", np.array([text.encode("utf-8")], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
