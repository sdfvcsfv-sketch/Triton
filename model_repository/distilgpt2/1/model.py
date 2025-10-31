import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import pipeline

class TritonPythonModel:
    def initialize(self, args):
        self.gen = pipeline("text-generation", model="distilgpt2", device=-1)  # CPU

    def execute(self, requests):
        responses = []
        for request in requests:
            
            input_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt_bytes = input_tensor.as_numpy()[0]
            prompt = prompt_bytes.decode("utf-8")

            try:
                out_text = self.gen(prompt, max_length=50, do_sample=False)[0]["generated_text"]
            except Exception as e:
                out_text = f"ERROR: {e}"

            # TYPE_STRING -> bytes in a numpy.object_ array
            out = np.array([out_text.encode("utf-8")], dtype=object)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("completion", out)]
            ))
        return responses
