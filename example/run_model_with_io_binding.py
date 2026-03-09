# Run a model using I/O binding (avoids copies when using device memory)
import ortpy as ort
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Run an ONNX model using I/O binding.")
parser.add_argument("--model_path", "-m", type=Path, required=True, help="Path to the ONNX model file.")
args = parser.parse_args()

model_path = Path(args.model_path)
session_options = ort.SessionOptions()
session = ort.Session(str(model_path), session_options)

input_info = session.get_input_info()
for input_name, tensor_info in input_info.items():
    print(f"Input {input_name}: {tensor_info.shape} {tensor_info.dimensions} {tensor_info.dtype}")
    if any(dim <= 0 for dim in tensor_info.shape):
        print("Model has non-positive input shapes. It is not run.")
        exit(0)

# Create I/O binding
binding = session.create_io_binding()

# Bind inputs as OrtValues
for input_name, tensor_info in input_info.items():
    data = np.random.uniform(low=0, high=1, size=tuple(tensor_info.shape)).astype(tensor_info.dtype)
    binding.bind_input(input_name, ort.Value(data))

# Bind outputs to CPU (ORT allocates the output buffers)
cpu_mem = ort.MemoryInfo()
output_info = session.get_output_info()
for output_name in output_info:
    binding.bind_output_to_device(output_name, cpu_mem)

# Run with binding
session.run_with_binding(binding)

# Retrieve outputs
outputs = binding.get_outputs()
print(f"Model run completed")
for output_name, output in outputs.items():
    arr = output.numpy()
    print(f"Output {output_name}: shape={arr.shape} dtype={arr.dtype}")
