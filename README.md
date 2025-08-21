# onnxruntime.py

## Build steps

### Prerequisites

1. Prepare a shell with Visual Studio 2022 tools and python enabled.
2. [optional]Create a new python environment.
3. `pip install -r requirements.txt`
4. `GetOnnxruntime.ps1`

### Build wheels

```PowerShell
mkdir build
cd build
cmake ..
cmake --build .
```

## Example

Please install/reinstall the new wheels before running the examples.

Please find examples under `./examples`
