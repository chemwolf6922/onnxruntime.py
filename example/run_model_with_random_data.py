# Run a model with random inputs
import pyort as ort
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Run an ONNX model with random data.")
parser.add_argument("--model_path", "-m", type=Path, required=True, help="Path to the ONNX model file.")
args = parser.parse_args()

model_path = Path(args.model_path)
session_options = ort.SessionOptions()

session = ort.Session(str(model_path), session_options)
input_info = session.get_input_info()
inputs = {}
has_non_positive_dim = False
for input_name, tensor_info in input_info.items():
    print(f"Input {input_name}: {tensor_info.shape} {tensor_info.dimensions} {tensor_info.dtype}")
    if any(dim <=0 for dim in tensor_info.shape):
        has_non_positive_dim = True
    else:
        inputs[input_name] = np.random.uniform(low=0 ,high=1, size=tuple(tensor_info.shape)).astype(tensor_info.dtype)
if has_non_positive_dim:
    print("Model has non-positive input shapes. It is not run.")
    exit(0)
outputs = session.run(inputs)
print(f"Model run completed")
for output_name, output in outputs.items():
    print(f"Output {output_name}: {output.shape} {output.dtype}")
