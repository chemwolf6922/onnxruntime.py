# onnxruntime-py

Minimal pybind11 native Python package that links to the ONNX Runtime CPU x64 distribution.

## Prerequisites
- Windows with MSVC Build Tools (Visual Studio or Build Tools)
- Python 3.8+
- CMake >= 3.20, Ninja (installed automatically via pip when building)
- ONNX Runtime CPU x64 extracted under `./onnxruntime` (run `GetOnnxruntime.ps1`)

## Build

PowerShell:

```powershell
# Prepare ONNX Runtime
./GetOnnxruntime.ps1

# Build wheel
py -m pip install -U pip
py -m pip install -U build
py -m build

# Install editable (optional)
py -m pip install -e .

# Test
python -c "import pyort as p; import pyort._pyort as m; print(p.__version__, m.hello())"
```

The build uses scikit-build-core + CMake to compile the extension and links against `onnxruntime.lib`.
