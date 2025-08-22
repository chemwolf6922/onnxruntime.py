# Run a model with random inputs
import pyort as ort
import numpy as np
from pathlib import Path
import argparse
import time

parser = argparse.ArgumentParser(description="Run an ONNX model with random data.")
parser.add_argument("--model_path", "-m", type=Path, required=True, help="Path to the ONNX model file.")
parser.add_argument("--ep", "-e", type=str, required=False, help="Execution provider to use, e.g. WebGpuExecutionProvider.")
args = parser.parse_args()

model_path = Path(args.model_path)
session_options = ort.SessionOptions()

if args.ep:
    print('looking for EP', args.ep)
    ep_devices = ort.get_ep_devices()
    for ep_device in ep_devices:
        if ep_device.ep_name == args.ep:
            session_options.append_execution_provider_v2([ep_device], {})
            break

session = ort.Session(str(model_path), session_options)
input_info = session.get_input_info()
inputs = {}
for input_name, tensor_info in input_info.items():
    inputs[input_name] = np.random.uniform(low=0 ,high=1, size=tuple(tensor_info.shape)).astype(tensor_info.dtype)
start = time.monotonic()
outputs = session.run(inputs)
duration = time.monotonic() - start
print(f"Model run completed in {duration:.4f} seconds.")
for output_name, output in outputs.items():
    print(f"Output {output_name}: {output.shape} {output.dtype}")
