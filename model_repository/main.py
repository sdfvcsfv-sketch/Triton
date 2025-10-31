import tritonclient.http as http
import numpy as np
from PIL import Image
import io
import os
client = http.InferenceServerClient("localhost:8000")


# ---- OCR ----
img = Image.open("test.png").convert("RGB")
buf = io.BytesIO(); img.save(buf, format="PNG")
arr = np.array([buf.getvalue()], dtype=object)

inp = http.InferInput("input_image", [1], "BYTES")
inp.set_data_from_numpy(arr)


inp.set_data_from_numpy(arr)

res = client.infer("ocr", inputs=[inp])
ocr_text = res.as_numpy("text")[0].decode()
print("OCR text:", ocr_text)

# ---- GPT ----
prompt_in = http.InferInput("prompt", [1,1], "BYTES")
prompt_in.set_data_from_numpy(np.array([[ocr_text.encode()]], dtype=object))

res2 = client.infer("distilgpt2", inputs=[prompt_in])
print("GPT completion:", res2.as_numpy("completion")[0].decode())
