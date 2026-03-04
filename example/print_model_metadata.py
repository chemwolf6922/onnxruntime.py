# Print model metadata from an ONNX model file
import ortpy as ort
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Print metadata of an ONNX model.")
parser.add_argument("--model_path", "-m", type=Path, required=True, help="Path to the ONNX model file.")
args = parser.parse_args()

model_path = Path(args.model_path)
session_options = ort.SessionOptions()
session = ort.Session(str(model_path), session_options)

metadata = session.get_model_metadata()

print(f"Producer:          {metadata.producer_name}")
print(f"Graph name:        {metadata.graph_name}")
print(f"Domain:            {metadata.domain}")
print(f"Description:       {metadata.description}")
print(f"Graph description: {metadata.graph_description}")
print(f"Version:           {metadata.version}")

custom = metadata.custom_metadata_map
if custom:
    print(f"Custom metadata ({len(custom)} entries):")
    for key, value in custom.items():
        print(f"  {key}: {value}")
else:
    print("Custom metadata:   (none)")
